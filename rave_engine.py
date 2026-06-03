"""Streaming RAVE TorchScript wrapper.

Contract:
    RAVEEngine(model_path)
        .latent_dim: int        # introspected from the loaded model
        .sample_rate: int       # introspected from the loaded model
        .hop_size: int          # audio samples produced per latent tick
        .decode(z: np.ndarray) -> np.ndarray  # 1D float32 mono audio

Load the streaming RAVE model once at module level (torch.jit.load). The
model carries its own cached-convolution state across calls.

Different RAVE exports expose their shape attributes inconsistently:

- Some have `model.sr` (int); others don't.
- `latent_size` sometimes means truncated PCA dim (8 for the bird model),
  sometimes the full latent (128 for the organ model). We use
  `decode_params[0]` instead, which is always the dim decode() expects.
- `decode()` itself takes either (z, scale_bool) or just (z) depending on
  whether the model was exported with the variational scaling argument.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch


class RAVEEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._model = torch.jit.load(str(self.model_path), map_location="cpu").eval()

        # decode_params is [latent_dim, hop, n_channels_out, ?] across both
        # the variational and trace exports we have seen.
        dp = getattr(self._model, "decode_params", None)
        if dp is not None:
            self.latent_dim: int = int(dp[0])
            self.hop_size: int = int(dp[1])
        else:
            self.latent_dim = int(getattr(self._model, "latent_size", 8))
            self.hop_size = 512

        # Sample rate: model attr if present, else parse from filename.
        sr_attr = getattr(self._model, "sr", None)
        if sr_attr is not None:
            self.sample_rate: int = int(sr_attr)
        else:
            m = re.search(r"_r(\d+)_", self.model_path.name)
            self.sample_rate = int(m.group(1)) if m else 48_000

        self._decode_takes_flag = self._detect_decode_signature()
        self._warmup()

    def _detect_decode_signature(self) -> bool:
        z = torch.zeros(1, self.latent_dim, 1)
        with torch.no_grad():
            try:
                self._model.decode(z, True)
                return True
            except (RuntimeError, TypeError):
                self._model.decode(z)
                return False

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return (
            self._model.decode(z, True)
            if self._decode_takes_flag
            else self._model.decode(z)
        )

    def _warmup(self, T: int = 2, n: int = 5) -> None:
        z = torch.zeros(1, self.latent_dim, T)
        with torch.no_grad():
            for _ in range(n):
                self._decode(z)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode one latent slice into audio.

        Args:
            z: shape [1, latent_dim, T], float32.
        Returns:
            1D float32 mono audio of length T * hop_size.
        """
        z_t = torch.from_numpy(np.ascontiguousarray(z)).float()
        with torch.no_grad():
            audio = self._decode(z_t)
        return audio[0, 0].numpy()
