"""Headless smoke test for swap.py — no webcam required.

Uses insightface's bundled sample image (two faces) as both source and
target, so it runs anywhere with network access (buffalo_l auto-downloads
on first run). Exercises the full path: detector load, swapper load,
swap_image (paste-back), and swap_video (synthesized clip -> libx264).

    uv run python scripts/smoke_test.py

Writes /tmp/smoke_image.jpg and /tmp/smoke_video.mp4 on success.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import imageio.v3 as iio
import numpy as np
from insightface.data import get_image as ins_get_image

from swap import FaceSwapper


def main() -> None:
    print("loading sample image (insightface 't1', two faces)...")
    rgb = ins_get_image("t1")  # RGB HxWx3
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Downscale to keep CPU swap quick for the smoke test. Force even
    # dimensions — libx264/yuv420p rejects odd width or height.
    h, w = bgr.shape[:2]
    scale = 640 / max(h, w)
    nw, nh = (int(w * scale) // 2) * 2, (int(h * scale) // 2) * 2
    small = cv2.resize(bgr, (nw, nh))

    print("loading FaceSwapper (buffalo_l auto-downloads on first run)...")
    t0 = time.time()
    swapper = FaceSwapper()
    print(f"  models ready in {time.time() - t0:.1f}s")

    # --- swap_image ---
    print("swap_image: self-swap on sample...")
    out_img = swapper.swap_image(small, small)
    if out_img is None:
        sys.exit("FAIL: swap_image returned None (no face detected)")
    cv2.imwrite("/tmp/smoke_image.jpg", out_img)
    print(f"  OK -> /tmp/smoke_image.jpg {out_img.shape}")

    # --- swap_video ---
    print("swap_video: synthesizing a 12-frame target clip...")
    target = Path("/tmp/smoke_target.mp4")
    writer = iio.imopen(target, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=12)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    for _ in range(12):
        writer.write_frame(rgb_small)
    writer.close()

    out_video = Path("/tmp/smoke_video.mp4")
    t0 = time.time()
    result = swapper.swap_video(small, target, out_video, keep_audio=False)
    if result is None:
        sys.exit("FAIL: swap_video returned None (no face in source)")
    dt = time.time() - t0
    print(f"  OK -> {result} (12 frames in {dt:.1f}s, ~{dt / 12:.2f}s/frame)")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
