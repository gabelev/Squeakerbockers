"""Streaming RAVE TorchScript wrapper.

Contract:
    RAVEEngine(model_path)
        .latent_dim: int        # introspected from the loaded model
        .sample_rate: int       # introspected from the loaded model
        .decode(z: np.ndarray) -> np.ndarray  # audio chunk

Load an exported *streaming* RAVE model with torch.jit.load. The model is
heavy: load once at module level, not inside the handler's copy(). The
exposed latent_dim / sample_rate come from the model, not config.
Implementation lands at P5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class RAVEEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._model = None  # torch.jit.load at P5
        self.latent_dim: int = 8
        self.sample_rate: int = 48_000

    def decode(self, z: np.ndarray) -> np.ndarray:
        # Placeholder silence until P5.
        return np.zeros(2048, dtype=np.float32)
