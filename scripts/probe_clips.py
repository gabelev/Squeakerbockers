"""Curate candidate target clips by how detectable the dunker's face is.

For each video passed (or every mp4 in assets/dunks/), sample frames and
report: dimensions, fps, duration, % of sampled frames with a detected
face, and the median face width as a fraction of frame width (a proxy for
swap quality — tiny faces swap badly).

    uv run python scripts/probe_clips.py [clip.mp4 ...]
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import imageio.v3 as iio
import numpy as np
from insightface.app import FaceAnalysis

SAMPLE_EVERY = 5  # probe 1 in N frames for speed


def probe(app: FaceAnalysis, path: Path) -> dict:
    meta = iio.immeta(str(path), plugin="pyav")
    fps = float(meta.get("fps", 30.0))
    widths: list[float] = []
    hits = 0
    sampled = 0
    dims = None
    for i, frame in enumerate(iio.imiter(str(path), plugin="pyav")):
        if i % SAMPLE_EVERY:
            continue
        sampled += 1
        if dims is None:
            dims = (frame.shape[1], frame.shape[0])  # (w, h)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if faces:
            hits += 1
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]))
            widths.append((f.bbox[2] - f.bbox[0]) / frame.shape[1])
    n_frames = i + 1
    return {
        "dims": dims,
        "fps": round(fps, 1),
        "frames": n_frames,
        "dur_s": round(n_frames / fps, 1),
        "face_rate": round(hits / sampled, 2) if sampled else 0.0,
        "median_face_w": round(median(widths), 3) if widths else 0.0,
    }


def main() -> None:
    clips = [Path(p) for p in sys.argv[1:]]
    if not clips:
        clips = sorted((Path(__file__).resolve().parent.parent / "assets/dunks").glob("*.mp4"))
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    print(f"\n{'clip':32} {'dims':>11} {'fps':>4} {'dur':>5} {'face%':>6} {'face_w':>7}  verdict")
    print("-" * 88)
    for c in clips:
        try:
            r = probe(app, c)
        except Exception as e:
            print(f"{c.name:32} ERROR {type(e).__name__}: {e}")
            continue
        # Heuristic verdict for swap suitability.
        ok_rate = r["face_rate"] >= 0.4
        ok_size = r["median_face_w"] >= 0.05
        verdict = "GOOD" if (ok_rate and ok_size) else ("weak-face" if ok_rate else "few/no faces")
        d = f'{r["dims"][0]}x{r["dims"][1]}' if r["dims"] else "?"
        print(f"{c.name:32} {d:>11} {r['fps']:>4} {r['dur_s']:>5} "
              f"{int(r['face_rate']*100):>5}% {r['median_face_w']:>7}  {verdict}")


if __name__ == "__main__":
    main()
