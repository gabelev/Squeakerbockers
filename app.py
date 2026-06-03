"""Squeakerbockers — Gradio entry point.

Phase 2b: FastRTC Stream wired to SqueakerHandler. Video passes through
(no overlay yet) and a steady 440 Hz sine wave streams out. Proves the
real handler shape works on the wire before pose, features, or RAVE.

See SPEC.md for the full architecture.
"""
from fastrtc import Stream

from handler import SqueakerHandler


def build_stream() -> Stream:
    return Stream(
        handler=SqueakerHandler(),
        modality="audio-video",
        mode="send-receive",
    )


if __name__ == "__main__":
    build_stream().ui.launch()
