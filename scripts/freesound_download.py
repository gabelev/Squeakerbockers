"""Download Freesound preview MP3s listed in a search CSV.

Usage:
    uv run python scripts/freesound_download.py shoe.csv data/raw/freesound/
    uv run python scripts/freesound_download.py court.csv data/raw/freesound/ --quality lq

Previews are MP3 128 kbps (hq) or ~64 kbps (lq). They don't need OAuth.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests


def safe_name(name: str, max_len: int = 80) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return out[:max_len] or "sound"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument(
        "--quality",
        choices=["hq", "lq"],
        default="hq",
        help="preview quality (hq = 128kbps mp3, lq = 64kbps mp3)",
    )
    args = p.parse_args()

    if not args.csv_path.exists():
        sys.exit(f"CSV not found: {args.csv_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.csv_path.open() as f:
        rows = list(csv.DictReader(f))

    url_key = f"preview_{args.quality}_mp3"
    print(f"Downloading {len(rows)} preview(s) ({args.quality}) -> {args.output_dir}")

    n_ok = 0
    n_skip = 0
    for i, row in enumerate(rows, 1):
        url = (row.get(url_key) or "").strip()
        if not url:
            n_skip += 1
            continue
        name = safe_name(row.get("name", ""))
        out_path = args.output_dir / f"{row['id']}_{name}.mp3"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_ok += 1
            continue
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            n_ok += 1
            print(f"  [{i}/{len(rows)}] {out_path.name}  ({len(r.content)//1024} KB)")
        except Exception as e:
            print(f"  FAIL {url}: {e}", file=sys.stderr)

    print(f"\nDone. {n_ok} files written, {n_skip} skipped.")


if __name__ == "__main__":
    main()
