"""FastRTC handler for Squeakerbockers.

Phase 5: emit() now decodes a fixed RAVE latent slice instead of a sine wave.
Pose still runs in video_receive, skeleton overlay returned via video_emit.

Remaining wiring per SPEC §3:
- P4: video_receive also writes state.features.
- P6: emit reads state.features, controller maps to latent, RAVE decodes.
"""
from __future__ import annotations

import asyncio

import numpy as np
from fastrtc import AsyncAudioVideoStreamHandler

from config import POSE_MODEL, RAVE_MODEL_PATH
from pose import PoseModel
from rave_engine import RAVEEngine

# Module-level heavy loads (survive handler.copy()).
_pose = PoseModel(POSE_MODEL)
_pose.warmup()
_rave = RAVEEngine(RAVE_MODEL_PATH)

# Latent ticks decoded per emit() call. T=2 -> 1024 samples at hop=512.
LATENT_TICKS_PER_EMIT = 2
# RAVE outputs sit at low amplitude (~±0.02 here); a fixed gain makes the
# placeholder audibly present without clipping.
OUTPUT_GAIN = 8.0


class SqueakerHandler(AsyncAudioVideoStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_rave.sample_rate,
            input_sample_rate=48_000,
            fps=30,
        )
        self._last_overlay: np.ndarray | None = None

    def copy(self) -> "SqueakerHandler":
        return SqueakerHandler()

    async def start_up(self) -> None:
        return None

    # --- video --------------------------------------------------------
    async def video_receive(self, frame: np.ndarray) -> None:
        # YOLO is CPU-bound; off-thread so audio cadence is not blocked.
        _, annotated = await asyncio.to_thread(_pose.detect_with_overlay, frame)
        self._last_overlay = annotated

    async def video_emit(self) -> np.ndarray:
        if self._last_overlay is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self._last_overlay

    # --- audio --------------------------------------------------------
    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        return None  # mic input unused

    async def emit(self) -> tuple[int, np.ndarray]:
        # P5: decode a fixed nonzero latent so there is audible output to
        # verify against. Drives the first two PCA-aligned latent dims;
        # P6 replaces these constants with feature-driven values.
        z = np.zeros((1, _rave.latent_dim, LATENT_TICKS_PER_EMIT), dtype=np.float32)
        z[0, 0, :] = 1.5
        z[0, 1, :] = -0.5
        audio = _rave.decode(z) * OUTPUT_GAIN
        audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return (_rave.sample_rate, audio_i16)
