"""FastRTC handler for Squeakerbockers.

Phase 6 (end-to-end): video_receive runs pose + features and writes
state.features. emit reads state.features, runs the controller, decodes
through RAVE, returns the audio chunk. video_emit returns the latest
skeleton overlay. The two loops run on independent FastRTC cadences and
communicate through state.py.
"""
from __future__ import annotations

import asyncio

import numpy as np
from fastrtc import AsyncAudioVideoStreamHandler

from config import POSE_CONF, POSE_IMGSZ, POSE_MODEL, RAVE_MODEL_PATH
from control import Controller
from features import FeatureExtractor
from pose import PoseModel
from rave_engine import RAVEEngine
from state import state

# Module-level heavy loads (survive handler.copy()).
_pose = PoseModel(POSE_MODEL)
_pose.warmup(h=POSE_IMGSZ, w=POSE_IMGSZ)
_rave = RAVEEngine(RAVE_MODEL_PATH)
_features = FeatureExtractor()
_controller = Controller(_rave.latent_dim)

# WebRTC audio frame: 20 ms at the model's sample rate (per fastrtc's
# AUDIO_PTIME = 0.02s). We pop exactly this many samples per emit so we
# never overproduce relative to what fastrtc consumes.
WEBRTC_FRAME = int(_rave.sample_rate * 0.02)

# RAVE outputs amplitude varies by model; gain brings them into a comfortable
# level. Organ is much louder than bird (more harmonic energy at the same z).
# tanh is applied after gain so loud peaks saturate cleanly rather than crackling.
OUTPUT_GAIN = 3.0


class SqueakerHandler(AsyncAudioVideoStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_rave.sample_rate,
            input_sample_rate=48_000,
            fps=30,
        )
        self._last_overlay: np.ndarray | None = None
        self._pose_busy: bool = False
        self._frame_log_count: int = 0
        # Audio ring buffer: emit decodes only when this is short of one
        # WebRTC frame, otherwise pops the next frame and returns. Decouples
        # RAVE's hop size from WebRTC's frame size.
        self._audio_buf: np.ndarray = np.zeros(0, dtype=np.float32)

    def copy(self) -> "SqueakerHandler":
        return SqueakerHandler()

    async def start_up(self) -> None:
        return None

    # --- video --------------------------------------------------------
    async def video_receive(self, frame: np.ndarray) -> None:
        if self._pose_busy:
            return
        self._pose_busy = True
        try:
            detections, annotated = await asyncio.to_thread(
                _pose.detect_with_overlay, frame, POSE_CONF, POSE_IMGSZ
            )
            feats = _features(detections, frame_width=annotated.shape[1])
            state.features = feats
            self._last_overlay = annotated

            self._frame_log_count += 1
            if self._frame_log_count % 30 == 0:
                print(
                    f"[features] persons={feats.person_count} "
                    f"motion={feats.motion_energy:6.1f} "
                    f"foot_v={feats.foot_velocity:6.1f} "
                    f"pivot={feats.pivot_sharpness:6.1f} "
                    f"pan={feats.pan:.2f} prox={feats.proximity:6.0f}",
                    flush=True,
                )
        finally:
            self._pose_busy = False

    async def video_emit(self) -> np.ndarray:
        if self._last_overlay is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self._last_overlay

    # --- audio --------------------------------------------------------
    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        return None  # mic input unused

    async def emit(self) -> tuple[int, np.ndarray]:
        if len(self._audio_buf) < WEBRTC_FRAME:
            deficit = WEBRTC_FRAME - len(self._audio_buf)
            T = max(1, (deficit + _rave.hop_size - 1) // _rave.hop_size)
            z = _controller.step(state.features, T=T)
            new_audio = _rave.decode(z) * OUTPUT_GAIN
            self._audio_buf = np.concatenate([self._audio_buf, new_audio])
        out = self._audio_buf[:WEBRTC_FRAME]
        self._audio_buf = self._audio_buf[WEBRTC_FRAME:]
        # Soft saturation: tanh keeps the waveform shaped instead of hard-clipping.
        audio_i16 = (np.tanh(out) * 32767).astype(np.int16)
        return (_rave.sample_rate, audio_i16)
