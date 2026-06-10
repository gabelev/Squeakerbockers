"""Grab a webcam still + run a face swap on a target video, in one go.

Usage:
    uv run python scripts/quick_swap.py <target_video> [<out_video>]

Default output: /tmp/swap-test.mp4
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from swap import FaceSwapper


def grab_webcam_still(warmup_frames: int = 10) -> "cv2.Mat":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("could not open webcam (camera 0). Grant Terminal camera access in System Settings.")
    # First few frames are often dark / auto-exposing.
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.05)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("webcam returned no frame")
    return frame


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: quick_swap.py <target_video> [<out_video>]")
        sys.exit(1)
    target = Path(sys.argv[1])
    if not target.exists():
        sys.exit(f"target video not found: {target}")
    out = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("/tmp/swap-test.mp4")

    print("grabbing webcam still...")
    source = grab_webcam_still()
    src_path = Path("/tmp/swap-source.jpg")
    cv2.imwrite(str(src_path), source)
    print(f"saved source: {src_path} ({source.shape[1]}x{source.shape[0]})")

    print("loading face swap models...")
    swapper = FaceSwapper()

    print(f"swapping {target.name} → {out}")
    result = swapper.swap_video(source, target, out)
    if result is None:
        sys.exit("swap failed — no face detected in webcam still.")
    print(f"done: {result}")


if __name__ == "__main__":
    main()
