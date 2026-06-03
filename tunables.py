"""Runtime-mutable tuning knobs.

The handler and controller import `t` and read its attributes each call,
so UI sliders can mutate these values live without restarting Python.
Defaults pull from config.py so editing config still sets the cold-start
baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import (
    LATENT_SMOOTH_ALPHA,
    MOTION_GAIN,
    PIVOT_GAIN,
)

_DEFAULT_OUTPUT_GAIN = 3.0


@dataclass
class _Tunables:
    pivot_gain: float = PIVOT_GAIN
    motion_gain: float = MOTION_GAIN
    smooth_alpha: float = LATENT_SMOOTH_ALPHA
    output_gain: float = _DEFAULT_OUTPUT_GAIN


t = _Tunables()
