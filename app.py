"""Squeakerbockers — Gradio entry point.

Full-screen FastRTC stream (the layout Gabe likes) with a right-side
collapsible Sidebar carrying live sound-design sliders. Sliders mutate
the `tunables.t` object the controller and handler read each call, so
tuning happens without restarting Python.

See SPEC.md for the full architecture.
"""
import gradio as gr
from fastrtc import Stream, WebRTC

from handler import SqueakerHandler
from tunables import t


_CSS = """
/* Keep the Gradio footer visible at the bottom but click-through so the
   WebRTC record button (full-screen mode) stays reachable. */
footer {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 0 !important;
    pointer-events: none !important;
    text-align: center !important;
}
footer a, footer button { pointer-events: auto !important; }
"""


def build_demo() -> gr.Blocks:
    stream = Stream(
        handler=SqueakerHandler(),
        modality="audio-video",
        mode="send-receive",
    )

    with gr.Blocks(title="Squeakerbockers", css=_CSS) as demo:
        webrtc = WebRTC(
            label="Stream",
            rtc_configuration=stream.rtc_configuration,
            track_constraints=stream.track_constraints,
            mode="send-receive",
            modality="audio-video",
        )
        stream.webrtc_component = webrtc

        with gr.Sidebar(label="Tuning", open=False, position="right", width=320):
            gr.Markdown("### Sound design\nLive — no restart needed.")
            pivot = gr.Slider(
                0.0, 10.0, value=t.pivot_gain, step=0.1,
                label="Pivot gain (screech)",
            )
            motion = gr.Slider(
                0.0, 10.0, value=t.motion_gain, step=0.1,
                label="Motion gain",
            )
            alpha = gr.Slider(
                0.0, 0.95, value=t.smooth_alpha, step=0.01,
                label="Smoothing (lower = snappier)",
            )
            output = gr.Slider(
                0.0, 10.0, value=t.output_gain, step=0.1,
                label="Output gain",
            )

            pivot.change(lambda v: setattr(t, "pivot_gain", v), inputs=pivot)
            motion.change(lambda v: setattr(t, "motion_gain", v), inputs=motion)
            alpha.change(lambda v: setattr(t, "smooth_alpha", v), inputs=alpha)
            output.change(lambda v: setattr(t, "output_gain", v), inputs=output)

        webrtc.stream(
            fn=stream.event_handler,
            inputs=[webrtc],
            outputs=[webrtc],
            time_limit=stream.time_limit,
            concurrency_limit=stream.concurrency_limit,
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
