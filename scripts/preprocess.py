"""Preprocess raw audio for RAVE training.

Walks an input directory, resamples every audio file it can decode to
mono at the target sample rate, optionally trims silence and normalizes
peaks, and writes clean WAVs to the output directory. Output is ready
for `rave preprocess`.

Usage:
    uv run python scripts/preprocess.py data/raw/ data/clean/
    uv run python scripts/preprocess.py data/raw/ data/clean/ --sr 48000 --no-trim
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}


def process_file(
    input_path: Path,
    output_path: Path,
    sr: int,
    trim_top_db: float | None,
    peak_norm_db: float | None,
) -> float | None:
    """Returns duration in seconds on success, None on skip."""
    try:
        audio, _ = librosa.load(str(input_path), sr=sr, mono=True)
    except Exception as e:
        print(f"  SKIP {input_path.name}: {e}", file=sys.stderr)
        return None

    if trim_top_db is not None:
        audio, _ = librosa.effects.trim(audio, top_db=trim_top_db)

    if len(audio) < sr * 0.5:
        print(f"  SKIP {input_path.name}: too short after trim", file=sys.stderr)
        return None

    if peak_norm_db is not None:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            target_peak = 10.0 ** (peak_norm_db / 20.0)
            audio = audio * (target_peak / peak)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio.astype(np.float32), sr, subtype="PCM_16")
    return float(len(audio)) / sr


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess audio for RAVE training")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--sr", type=int, default=48000, help="target sample rate")
    p.add_argument("--no-trim", action="store_true", help="skip silence trim")
    p.add_argument("--no-norm", action="store_true", help="skip peak normalize")
    p.add_argument("--trim-db", type=float, default=30.0)
    p.add_argument("--norm-db", type=float, default=-1.0)
    args = p.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"input dir does not exist: {args.input_dir}")

    files = sorted(
        f for f in args.input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    )
    print(f"Found {len(files)} audio file(s) under {args.input_dir}")
    if not files:
        return

    n_ok = 0
    total_sec = 0.0
    for i, f in enumerate(files, 1):
        rel = f.relative_to(args.input_dir).with_suffix(".wav")
        out = args.output_dir / rel
        dur = process_file(
            f, out,
            sr=args.sr,
            trim_top_db=None if args.no_trim else args.trim_db,
            peak_norm_db=None if args.no_norm else args.norm_db,
        )
        if dur is not None:
            n_ok += 1
            total_sec += dur
            print(f"  [{i}/{len(files)}] {rel}  ({dur:.1f}s)")

    print(
        f"\nDone. Wrote {n_ok}/{len(files)} files. "
        f"Total: {total_sec/60:.1f} min ({total_sec/3600:.2f} hr) of audio."
    )


if __name__ == "__main__":
    main()
