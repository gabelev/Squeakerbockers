"""Encode a real court audio clip through the trained model to get a
guaranteed on-manifold latent trajectory.

We save the trajectory to models/base_latent.npy. The controller reads
from it as a looping base and adds motion as perturbation on top.

Usage:
    uv run python scripts/encode_base_latent.py
    uv run python scripts/encode_base_latent.py --source path/to/clip.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import torch

from config import PROJECT_ROOT, RAVE_MODEL_PATH


def pick_default_source() -> Path:
    """Pick the largest WAV under data/clean — most material to loop on."""
    candidates = sorted(
        (PROJECT_ROOT / "data" / "clean").rglob("*.wav"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not candidates:
        sys.exit(
            "No WAVs in data/clean/. Pass --source explicitly or "
            "run scripts/preprocess.py first."
        )
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=None)
    p.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="cap on source duration to keep the .npy small (default 120s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "models" / "base_latent.npy",
    )
    args = p.parse_args()

    source = args.source or pick_default_source()
    print(f"source: {source}")

    model = torch.jit.load(str(RAVE_MODEL_PATH), map_location="cpu").eval()
    sr = int(getattr(model, "sr", 48000))

    audio, _ = librosa.load(str(source), sr=sr, mono=True)
    max_samples = int(args.max_seconds * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    print(f"loaded {len(audio)/sr:.1f}s at {sr} Hz")

    # encode wants [batch, channels, samples]
    x = torch.from_numpy(audio).float().reshape(1, 1, -1)
    with torch.no_grad():
        z = model.encode(x)  # [1, latent_dim, T]
    z_np = z.squeeze(0).numpy().astype(np.float32)  # [latent_dim, T]

    print(
        f"latent trajectory shape={z_np.shape} "
        f"(latent_dim={z_np.shape[0]}, T={z_np.shape[1]}, "
        f"~{z_np.shape[1] * 2048 / sr:.1f}s) "
        f"mean={z_np.mean():+.3f} std={z_np.std():.3f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, z_np)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
