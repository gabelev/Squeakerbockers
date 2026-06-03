"""Streaming RAVE TorchScript wrapper.

Contract:
    RAVEEngine(model_path)
        .latent_dim: int        # introspected from the loaded model
        .sample_rate: int       # introspected from the loaded model
        .hop_size: int          # audio samples produced per latent tick
        .decode(z: np.ndarray) -> np.ndarray  # 1D float32 mono audio

Load the streaming RAVE model once at module level (torch.jit.load). The
model carries its own cached-convolution state across calls, so successive
decode() calls produce continuous audio without buffer juggling on our side.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class RAVEEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._model = torch.jit.load(str(self.model_path), map_location="cpu").eval()

        # Introspect; do not hardcode.
        self.sample_rate: int = int(getattr(self._model, "sr", 48_000))
        self.latent_dim: int = int(getattr(self._model, "latent_size", 8))
        # encode_params is [n_channels_in, ratio_in, latent_dim, hop_size]
        ep = getattr(self._model, "encode_params", None)
        self.hop_size: int = int(ep[-1]) if ep is not None else 512

        # Pay JIT / cached-conv compile cost up front so the first realtime
        # decode doesn't burn 100-200ms.
        self._warmup()

    def _warmup(self, T: int = 2, n: int = 5) -> None:
        z = torch.zeros(1, self.latent_dim, T)
        with torch.no_grad():
            for _ in range(n):
                self._model.decode(z, True)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode one latent slice into audio.

        Args:
            z: shape [1, latent_dim, T], float32.
        Returns:
            1D float32 mono audio of length T * hop_size.
        """
        z_t = torch.from_numpy(np.ascontiguousarray(z)).float()
        with torch.no_grad():
            audio = self._model.decode(z_t, True)
        # audio: [1, 1, T*hop]
        return audio[0, 0].numpy()
