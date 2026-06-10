"""Face swap with insightface + inswapper_128.

Loads detector (`buffalo_l`) and swapper (`inswapper_128.onnx`) at module
init. Exposes two operations:

    FaceSwapper().swap_image(source_img, target_img) -> swapped_img
    FaceSwapper().swap_video(source_img, target_video, out_video) -> out_video

`source_img` is the face we want to paste *onto* the target (i.e. the
webcam grab of the user). `target_img/video` contains the face we're
replacing (the basketball player).

Models live in models/. buffalo_l is auto-downloaded by insightface on
first .prepare() call to ~/.insightface/. inswapper_128.onnx must be
present at models/inswapper_128.onnx (see README/download step).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v3 as iio
import insightface
import numpy as np
from insightface.app import FaceAnalysis

PROJECT_ROOT = Path(__file__).parent.resolve()
SWAPPER_PATH = PROJECT_ROOT / "models" / "inswapper_128.onnx"


class FaceSwapper:
    def __init__(self, det_size: tuple[int, int] = (640, 640)) -> None:
        # buffalo_l: detection + recognition + landmarks. CPU is fine for ~30
        # frames; if you have CUDA/CoreML providers swap ctx_id below.
        self.app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.swapper = insightface.model_zoo.get_model(
            str(SWAPPER_PATH), providers=["CPUExecutionProvider"]
        )

    def _largest_face(self, img: np.ndarray):
        """Pick the face with the biggest bbox — most likely the subject."""
        faces = self.app.get(img)
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    def swap_image(
        self, source_img: np.ndarray, target_img: np.ndarray
    ) -> np.ndarray | None:
        src = self._largest_face(source_img)
        tgt = self._largest_face(target_img)
        if src is None or tgt is None:
            return None
        return self.swapper.get(target_img, tgt, src, paste_back=True)

    def swap_video(
        self,
        source_img: np.ndarray,
        target_video: Path,
        out_video: Path,
        keep_audio: bool = True,
    ) -> Path | None:
        """Swap the source face into every frame of target_video.

        Frames where no face is detected pass through unchanged. Audio is
        muxed back from the source video at the end if keep_audio=True
        and ffmpeg is on PATH.
        """
        src_face = self._largest_face(source_img)
        if src_face is None:
            return None

        # imageio for read/write — handles odd codecs better than cv2.
        meta = iio.immeta(str(target_video), plugin="pyav")
        fps = float(meta.get("fps", 30.0))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            silent_path = Path(tmp.name)

        writer = iio.imopen(silent_path, "w", plugin="pyav")
        # Container will be inferred from suffix; codec defaults to libx264
        writer.init_video_stream("libx264", fps=fps)

        try:
            for frame in iio.imiter(str(target_video), plugin="pyav"):
                # imageio gives RGB; insightface/cv2 expect BGR.
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                tgt = self._largest_face(bgr)
                if tgt is not None:
                    bgr = self.swapper.get(bgr, tgt, src_face, paste_back=True)
                writer.write_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            writer.close()

        if keep_audio and shutil.which("ffmpeg"):
            self._mux_audio(silent_path, target_video, out_video)
            silent_path.unlink(missing_ok=True)
        else:
            shutil.move(str(silent_path), out_video)

        return out_video

    @staticmethod
    def _mux_audio(
        silent_video: Path, audio_source: Path, out: Path
    ) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(silent_video),
            "-i", str(audio_source),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    # Tiny self-test entry point: `uv run python swap.py source.jpg target.mp4 out.mp4`
    import sys

    if len(sys.argv) != 4:
        print("usage: swap.py <source_image> <target_video> <out_video>")
        sys.exit(1)
    src_path, tgt_path, out_path = (Path(p) for p in sys.argv[1:4])
    source = cv2.imread(str(src_path))
    if source is None:
        sys.exit(f"could not read source image: {src_path}")
    swapper = FaceSwapper()
    result = swapper.swap_video(source, tgt_path, out_path)
    print("wrote" if result else "failed (no face detected in source?)", result)
