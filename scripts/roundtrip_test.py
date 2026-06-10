"""Encode→decode round-trip test in one process.

If the trained streaming-exported model can faithfully reconstruct a real
audio clip in this script, encoding works. If even this sounds like
noise, the encode method on the streaming export is broken (different
from the encoder we listened to in TensorBoard).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf
import torch

from config import PROJECT_ROOT, RAVE_MODEL_PATH

SOURCE = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "freesound"
    / "376823_squeaky_shoe.wav.wav"
)


def main() -> None:
    model = torch.jit.load(str(RAVE_MODEL_PATH), map_location="cpu").eval()
    sr = int(getattr(model, "sr", 48000))

    # Load first 5 seconds of source
    audio, _ = librosa.load(str(SOURCE), sr=sr, mono=True, duration=5.0)
    print(
        f"original: {len(audio)/sr:.1f}s peak={np.max(np.abs(audio)):.3f} "
        f"rms={np.sqrt(np.mean(audio**2)):.3f}"
    )
    sf.write("/tmp/roundtrip-original.wav", audio, sr)

    x = torch.from_numpy(audio).float().reshape(1, 1, -1)

    # Bulk encode (current approach)
    with torch.no_grad():
        z_bulk = model.encode(x)
        recon_bulk = model.decode(z_bulk)
    recon_bulk_np = recon_bulk.squeeze().numpy()
    print(
        f"bulk recon: shape={recon_bulk_np.shape} "
        f"peak={np.max(np.abs(recon_bulk_np)):.3f} "
        f"rms={np.sqrt(np.mean(recon_bulk_np**2)):.3f}"
    )
    sf.write("/tmp/roundtrip-bulk.wav", recon_bulk_np, sr)

    # Chunked encode (8192 sample chunks — what the streaming model was tested at)
    # Fresh model state for this run
    model2 = torch.jit.load(str(RAVE_MODEL_PATH), map_location="cpu").eval()
    chunk = 8192
    z_chunks = []
    with torch.no_grad():
        for i in range(0, x.shape[-1] - chunk + 1, chunk):
            z_chunks.append(model2.encode(x[..., i : i + chunk]))
    z_streamed = torch.cat(z_chunks, dim=-1)
    print(f"streamed latents shape: {z_streamed.shape}")
    # Decode through a fresh model to avoid leftover encoder state
    model3 = torch.jit.load(str(RAVE_MODEL_PATH), map_location="cpu").eval()
    with torch.no_grad():
        recon_streamed = model3.decode(z_streamed)
    recon_streamed_np = recon_streamed.squeeze().numpy()
    print(
        f"streamed recon: shape={recon_streamed_np.shape} "
        f"peak={np.max(np.abs(recon_streamed_np)):.3f} "
        f"rms={np.sqrt(np.mean(recon_streamed_np**2)):.3f}"
    )
    sf.write("/tmp/roundtrip-streamed.wav", recon_streamed_np, sr)

    print(
        "\nFiles to compare:\n"
        "  /tmp/roundtrip-original.wav  (the source clip)\n"
        "  /tmp/roundtrip-bulk.wav      (encode whole clip then decode)\n"
        "  /tmp/roundtrip-streamed.wav  (encode 8192-sample chunks then decode)\n"
    )


if __name__ == "__main__":
    main()
