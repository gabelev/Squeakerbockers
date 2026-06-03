"""Map features -> next RAVE latent slice.

Contract:
    Controller(latent_dim).step(features, T) -> np.ndarray [1, latent_dim, T]

Drives the top few PCA latent dimensions from features (pivot_sharpness and
motion_energy as the loudest knobs), fills the rest with smoothed low-amp
noise, and smooths the trajectory over time to avoid zipper artifacts. This
is the sound-design layer; iterate by ear at P7.
"""
from __future__ import annotations

import numpy as np

from state import Features


class Controller:
    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self._last_z = np.zeros((1, latent_dim, 1), dtype=np.float32)

    def step(self, features: Features, T: int = 1) -> np.ndarray:
        return np.zeros((1, self.latent_dim, T), dtype=np.float32)
