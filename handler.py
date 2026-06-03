"""FastRTC handler for Squeakerbockers.

Phase 2b: pass-through video + fixed sine wave audio. Proves the
AsyncAudioVideoStreamHandler shape on the wire before pose, features,
RAVE, or control land. Each later phase replaces one method:

- P3: video_receive runs pose, writes state.overlay_frame
- P4: video_receive also writes state.features
- P5: emit calls RAVEEngine.decode instead of sin()
- P6: emit reads state.features, controller maps to latent, RAVE decodes
"""
from __future__ import annotations

import math

import numpy as np
from fastrtc import AsyncAudioVideoStreamHandler

OUTPUT_SAMPLE_RATE = 48_000  # matches the placeholder RAVE model (z8, r48000)
SINE_FREQ_HZ = 440.0         # A4, easy to confirm by ear
SINE_AMPLITUDE = 0.2         # leaves headroom
SINE_CHUNK = 960             # 20 ms at 48 kHz, matches WebRTC framing


class SqueakerHandler(AsyncAudioVideoStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            output_frame_size=SINE_CHUNK,
            input_sample_rate=48_000,
            fps=30,
        )
        self._last_frame: np.ndarray | None = None
        self._sine_n: int = 0  # running sample index for phase continuity

    def copy(self) -> "SqueakerHandler":
        return SqueakerHandler()

    async def start_up(self) -> None:
        return None

    # --- video --------------------------------------------------------
    async def video_receive(self, frame: np.ndarray) -> None:
        self._last_frame = frame

    async def video_emit(self) -> np.ndarray:
        if self._last_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        f = self._last_frame
        if f.dtype == np.float32:
            scale = 255.0 if f.max() <= 1.0 else 1.0
            f = np.clip(f * scale, 0, 255).astype(np.uint8)
        return f

    # --- audio --------------------------------------------------------
    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        return None  # mic input unused

    async def emit(self) -> tuple[int, np.ndarray]:
        n0 = self._sine_n
        ts = np.arange(n0, n0 + SINE_CHUNK) / OUTPUT_SAMPLE_RATE
        wave = SINE_AMPLITUDE * np.sin(2 * math.pi * SINE_FREQ_HZ * ts)
        self._sine_n = n0 + SINE_CHUNK
        audio = (wave * 32767).astype(np.int16)
        return (OUTPUT_SAMPLE_RATE, audio)
