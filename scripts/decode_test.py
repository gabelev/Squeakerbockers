"""Decode 5s of audio from pure OU-noise latents and write to a WAV.

Diagnostic for separating model behavior from FastRTC plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from config import RAVE_MODEL_PATH
from control import Controller
from rave_engine import RAVEEngine
from state import Features


def main() -> None:
    e = RAVEEngine(RAVE_MODEL_PATH)
    c = Controller(e.latent_dim)
    f = Features(
        person_count=0,
        motion_energy=0.0,
        foot_velocity=0.0,
        pivot_sharpness=0.0,
        pan=0.0,
        proximity=0.0,
    )
    chunks = []
    # ~5s of audio: hop_size 2048 at 48k = 23 Hz latent rate
    for _ in range(60):
        z = c.step(f, T=2)
        chunks.append(e.decode(z))
    audio = np.concatenate(chunks)
    out_path = "/tmp/squeak-test.wav"
    sf.write(out_path, audio, e.sample_rate)
    dur = len(audio) / e.sample_rate
    peak = float(np.max(np.abs(audio)))
    print(f"wrote {out_path}, {dur:.1f}s, peak={peak:.3f}")


if __name__ == "__main__":
    main()
