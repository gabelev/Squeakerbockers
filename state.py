"""Shared state between video and audio loops.

Two single-slot, last-write-wins holders. CPython attribute assignment is
atomic for object references — no lock needed for whole-object swaps. The
audio loop and the video_emit path tolerate stale reads; they must never
block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Features:
    motion_energy: float = 0.0
    foot_velocity: float = 0.0
    pivot_sharpness: float = 0.0
    pan: float = 0.5
    proximity: float = 0.0
    person_count: int = 0


class SharedState:
    def __init__(self) -> None:
        self.features: Features = Features()
        self.overlay_frame: Optional[np.ndarray] = None


state = SharedState()
