"""Squeakerbockers (face-swap pivot) — Gradio entry point.

Flow: capture your face (webcam or upload) -> "Send me to the Garden" ->
your face is swapped onto a pre-collected basketball clip -> play it back.
This is the upload/process/playback shape (NOT realtime) because the swap
runs ~several seconds per frame on CPU; see README. The old realtime-audio
app lives in app.py and is unrelated.

Run locally:
    ./run_swap.sh          # sets LD_LIBRARY_PATH for the headless OpenCV/GL libs
or:
    LD_LIBRARY_PATH=~/.local/gl-shim uv run python app_swap.py

On a Hugging Face Space, packages.txt installs libgl1 so LD_LIBRARY_PATH is
unnecessary there.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg

from swap import FaceSwapper

PROJECT_ROOT = Path(__file__).parent.resolve()
DUNKS_DIR = PROJECT_ROOT / "assets" / "dunks"

# Keep demo latency sane: the CPU swap is ~several seconds per frame, so cap
# the target to a short, low-fps clip. Tune here.
MAX_SECONDS = 2.0
TARGET_FPS = 10

# Curated swap-quality targets (large face facing camera — see
# assets/dunks/README.md). Action b-roll clips with tiny faces are excluded.
TARGET_CLIPS = [
    DUNKS_DIR / "cand_5319070.mp4",
]

# Heavy model load once at import (survives across requests).
_swapper = FaceSwapper()
_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

DISCLOSURE = (
    "⚠️ **Face-swap disclosure.** Your captured face is swapped onto stock "
    "footage **locally** on this machine — no cloud API, nothing uploaded. "
    "Only use a face you have the right to use (ideally your own)."
)


def _available_targets() -> list[Path]:
    return [p for p in TARGET_CLIPS if p.exists()]


def _prepare_target(src: Path) -> Path:
    """Trim a target clip to MAX_SECONDS @ TARGET_FPS with even dimensions
    (libx264/yuv420p requires even w/h). Returns a temp mp4 path."""
    out = Path(tempfile.mkstemp(suffix=".mp4")[1])
    cmd = [
        _FFMPEG, "-y", "-i", str(src),
        "-t", str(MAX_SECONDS), "-r", str(TARGET_FPS),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-an", str(out),
    ]
    import subprocess
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def garden_swap_photo(source_rgb):
    """Single-frame swap onto the first frame of a target clip. Fast (~1 swap)."""
    if source_rgb is None:
        raise gr.Error("Capture or upload a face first.")
    targets = _available_targets()
    if not targets:
        raise gr.Error("No target clips available in assets/dunks/.")
    # Pull frame 1 of a random target as the swap canvas.
    target = random.choice(targets)
    import imageio.v3 as iio
    frame_rgb = next(iio.imiter(str(target), plugin="pyav"))
    src_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = _swapper.swap_image(src_bgr, tgt_bgr)
    if out_bgr is None:
        raise gr.Error("No face detected in your photo (or in the target). Try again, facing the camera.")
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def garden_swap_video(source_rgb, progress=gr.Progress()):
    """Swap your face onto a short target clip and return the video path."""
    if source_rgb is None:
        raise gr.Error("Capture or upload a face first.")
    targets = _available_targets()
    if not targets:
        raise gr.Error("No target clips available in assets/dunks/.")

    src_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    if _swapper._largest_face(src_bgr) is None:
        raise gr.Error("No face detected in your photo. Face the camera and try again.")

    progress(0.0, desc="Warming up the Garden...")
    target = _prepare_target(random.choice(targets))
    out_path = Path(tempfile.mkstemp(suffix=".mp4")[1])

    def _cb(done: int, total: int) -> None:
        progress(done / max(total, 1), desc=f"Swapping frame {done}/{total}")

    result = _swapper.swap_video(
        src_bgr, target, out_path, keep_audio=False, progress_cb=_cb
    )
    target.unlink(missing_ok=True)
    if result is None:
        raise gr.Error("Swap failed — no face detected in your photo.")
    return str(result)


_CSS = """
.gradio-container { background: #04102b; }
#title h1 { color: #F58426; letter-spacing: .5px; }
#title p { color: #cdd6e6; }
button.primary, .primary { background: #F58426 !important; border-color: #F58426 !important; }
.tabitem { border-color: #006BB6 !important; }
"""


def build_demo() -> gr.Blocks:
    theme = gr.themes.Base(primary_hue="orange", secondary_hue="blue")
    with gr.Blocks(title="Squeakerbockers — Garden Dunk", theme=theme, css=_CSS) as demo:
        with gr.Column(elem_id="title"):
            gr.Markdown("# 🏀 Squeakerbockers — Dunk at the Garden")
            gr.Markdown("Put *your* face on the highlight reel. New York forever.")
        gr.Markdown(DISCLOSURE)

        with gr.Row():
            source = gr.Image(
                sources=["webcam", "upload"], type="numpy", label="Your face",
                height=320,
            )
            with gr.Tabs():
                with gr.Tab("Video (the full dunk)"):
                    out_video = gr.Video(label="You, at the Garden", height=320)
                    go_video = gr.Button("Send me to the Garden 🏀", variant="primary")
                    go_video.click(garden_swap_video, inputs=source, outputs=out_video)
                    gr.Markdown(
                        f"_Swaps a ~{MAX_SECONDS:.0f}s clip at {TARGET_FPS}fps. "
                        "On CPU this takes a couple minutes — watch the progress bar._"
                    )
                with gr.Tab("Photo (instant)"):
                    out_photo = gr.Image(label="You, at the Garden", height=320)
                    go_photo = gr.Button("Snap me into the game 📸", variant="primary")
                    go_photo.click(garden_swap_photo, inputs=source, outputs=out_photo)

    return demo


if __name__ == "__main__":
    build_demo().launch()
