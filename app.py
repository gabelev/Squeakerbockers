"""Squeakerbockers — Gradio entry point.

Phase 2b: FastRTC Stream wired to SqueakerHandler. Video passes through
(no overlay yet) and a steady 440 Hz sine wave streams out. Proves the
real handler shape works on the wire before pose, features, or RAVE.

See SPEC.md for the full architecture.
"""
from fastrtc import Stream

from handler import SqueakerHandler


def build_stream() -> Stream:
    stream = Stream(
        handler=SqueakerHandler(),
        modality="audio-video",
        mode="send-receive",
        # full_screen=True (default) gives a viewport-filling video widget;
        # we fix the badge/record-button collision with CSS below.
    )
    # Pin Gradio's footer to the page bottom and make the strip itself
    # click-through, so the WebRTC record button overlaying full-screen
    # mode remains reachable. Links inside the footer stay clickable.
    stream.ui.css = (stream.ui.css or "") + """
    footer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 0 !important;
        pointer-events: none !important;
        text-align: center !important;
    }
    footer a, footer button {
        pointer-events: auto !important;
    }
    """
    return stream


if __name__ == "__main__":
    build_stream().ui.launch()
