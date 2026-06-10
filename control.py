"""Map features -> RAVE latent slice with motion-driven trajectory scrubbing.

Reads a pre-encoded base trajectory from models/base_latent.npy (on the
trained manifold) and uses motion features to *scrub through* the
trajectory rather than perturb it. Result: motion is responsible for
*when* sounds happen, not just colouring them.

Mapping:
    - motion_energy → scan rate. Stillness holds the position (slow drift);
      movement advances through the trajectory.
    - pivot_sharpness → discrete position jumps. A sharp pivot teleports
      the scan head to a new region of the trajectory, producing a
      different sound on each pivot.
    - small motion-driven delta on dim 0/1 on top, for colour.

If the base trajectory is missing we fall back to an OU random walk
(useful only as a sanity test — will not sound musical).

Contract:
    Controller(latent_dim).step(features, T) -> np.ndarray [1, latent_dim, T]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import (
    MOTION_ENERGY_SCALE,
    NOISE_DIMS_SCALE,
    PIVOT_SHARPNESS_SCALE,
    PROJECT_ROOT,
)
from state import Features
from tunables import t

_LATENT_CLIP = 3.0
_BASE_LATENT_PATH = PROJECT_ROOT / "models" / "base_latent.npy"

# Scan rate (in trajectory ticks per emit call) when standing perfectly still.
# Non-zero so the loop drifts instead of freezing on a single grain.
_MIN_SCAN_RATE = 0.05

# motion_energy multiplied by this becomes additional scan rate.
_MOTION_SCAN_GAIN = 0.4

# pivot_sharpness above this fraction of clip range triggers a position jump.
_PIVOT_JUMP_THRESHOLD = 0.5

# OU process for the noise fallback path.
_NOISE_RW_ALPHA = 0.98
_NOISE_STEP_GAIN = float(np.sqrt(1.0 - _NOISE_RW_ALPHA ** 2))


class Controller:
    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self._base = self._load_base_latent()
        self._pos: float = 0.0  # fractional position in trajectory ticks
        self._last_pivot: float = 0.0
        self._noise = np.zeros(latent_dim, dtype=np.float32)
        self._rng = np.random.default_rng()

    def _load_base_latent(self) -> np.ndarray | None:
        if not _BASE_LATENT_PATH.exists():
            return None
        arr = np.load(_BASE_LATENT_PATH).astype(np.float32)  # [latent_dim, T]
        if arr.shape[0] != self.latent_dim:
            print(
                f"[control] base_latent.npy latent_dim={arr.shape[0]} "
                f"!= model latent_dim={self.latent_dim}; ignoring."
            )
            return None
        print(
            f"[control] loaded base latent: shape={arr.shape}, "
            f"~{arr.shape[1] * 2048 / 48000:.1f}s of material to scrub"
        )
        return arr

    def step(self, features: Features, T: int = 2) -> np.ndarray:
        if self._base is None:
            return self._fallback_noise(T)

        T_long = self._base.shape[1]

        # Pivot triggers a jump to a random region of the trajectory.
        # Use a hysteresis-style check on the rising edge of pivot_sharpness
        # so a single sharp event causes one jump rather than continuous chaos.
        pivot_norm = min(
            abs(features.pivot_sharpness) * PIVOT_SHARPNESS_SCALE * t.pivot_gain,
            1.0,
        )
        if (
            pivot_norm > _PIVOT_JUMP_THRESHOLD
            and self._last_pivot <= _PIVOT_JUMP_THRESHOLD
        ):
            self._pos = float(self._rng.integers(0, T_long))
        self._last_pivot = pivot_norm

        # Scan rate driven by motion_energy. Stillness still drifts slowly.
        motion_norm = features.motion_energy * MOTION_ENERGY_SCALE * t.motion_gain
        scan_rate = _MIN_SCAN_RATE + max(0.0, motion_norm) * _MOTION_SCAN_GAIN

        # Pull T consecutive ticks starting at current position.
        idx = ((np.arange(T) * scan_rate) + self._pos).astype(np.int64) % T_long
        z = self._base[:, idx].copy()
        self._pos = (self._pos + T * scan_rate) % T_long

        # Optional small perturbation on dim 0 from sustained pivot pressure
        # (gives a continuous colouring when held high, not just on jumps).
        z[0, :] = np.clip(
            z[0, :] + pivot_norm * 0.5, -_LATENT_CLIP, _LATENT_CLIP
        )

        return z[None, :, :].astype(np.float32).copy()

    def _fallback_noise(self, T: int) -> np.ndarray:
        step = self._rng.standard_normal(self.latent_dim).astype(np.float32)
        self._noise = (
            _NOISE_RW_ALPHA * self._noise + _NOISE_STEP_GAIN * step
        ).astype(np.float32)
        z = np.broadcast_to(
            (self._noise * NOISE_DIMS_SCALE)[:, None], (self.latent_dim, T)
        ).copy()
        return z[None, :, :].astype(np.float32).copy()
