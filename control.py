"""Map features -> RAVE latent slice with smoothing.

Drives the first two latent dimensions from the loudest features
(pivot_sharpness and motion_energy), fills the rest with low-amp noise,
and smooths the trajectory across emit calls to suppress zipper noise.

Contract:
    Controller(latent_dim).step(features, T) -> np.ndarray [1, latent_dim, T]

The character of the sound lives here; tune by ear at P7.
"""
from __future__ import annotations

import numpy as np

from config import (
    LATENT_SMOOTH_ALPHA,
    MOTION_ENERGY_SCALE,
    MOTION_GAIN,
    NOISE_DIMS_SCALE,
    PIVOT_GAIN,
    PIVOT_SHARPNESS_SCALE,
)
from state import Features

_LATENT_CLIP = 3.0


class Controller:
    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self._z = np.zeros(latent_dim, dtype=np.float32)

    def step(self, features: Features, T: int = 2) -> np.ndarray:
        target = np.zeros(self.latent_dim, dtype=np.float32)
        target[0] = float(
            np.clip(
                features.pivot_sharpness * PIVOT_SHARPNESS_SCALE * PIVOT_GAIN,
                -_LATENT_CLIP,
                _LATENT_CLIP,
            )
        )
        if self.latent_dim > 1:
            target[1] = float(
                np.clip(
                    features.motion_energy * MOTION_ENERGY_SCALE * MOTION_GAIN,
                    -_LATENT_CLIP,
                    _LATENT_CLIP,
                )
            )
        # Non-driven dims left at 0 for now. Adding fresh np.random.randn() here
        # every emit (~50Hz) sounds like static after smoothing — a slow LFO is
        # the right fix if we want organic motion in P7. NOISE_DIMS_SCALE stays
        # in config for that future tuning.

        alpha = LATENT_SMOOTH_ALPHA
        self._z = (alpha * self._z + (1.0 - alpha) * target).astype(np.float32)

        z_slice = np.broadcast_to(
            self._z[None, :, None], (1, self.latent_dim, T)
        ).astype(np.float32).copy()
        return z_slice
