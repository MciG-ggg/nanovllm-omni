"""Top-level Gradio demo for ``nanovllm-omni``.

Composes four lazy-loaded tabs (audio / image / chat / vla) into a single
``gr.Blocks`` instance. Each tab module owns its own engine state and
never reloads the model within the session -- the 4 GB consumer-card
target can't hold more than one ``Omni(...)`` engine in VRAM at a time.

Run with::

    python -m nanovllm_omni.serving.app

Heavy deps are imported lazily inside each tab module so launching the
UI does not require every model family's optional extra to be installed
at once. Pre-install at least one of ``[minimind]``, ``[smolvla]``,
``[sd_turbo]``-style deps for the tab you intend to use.
"""

from __future__ import annotations

from typing import Any

from . import audio as _audio_tab
from . import chat as _chat_tab
from . import image as _image_tab
from . import vla as _vla_tab


def build_demo() -> Any:
    """Build the multi-tab Gradio demo. Returns without launching."""
    import gradio as gr  # noqa: PLC0415 -- lazy: avoid hard dep on gradio at import

    with gr.Blocks(title="nanovllm-omni demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# nanovllm-omni demo\n"
            "Each tab lazy-loads its own engine on first click; do not load more than one "
            "tab's model at once on a 4 GB GPU.\n\n"
            "Pre-install the matching extras (`[minimind]`, `[smolvla]`, ...) before "
            "launching; see the README for the per-tab weight snapshot commands."
        )
        _audio_tab.register(demo)
        _image_tab.register(demo)
        _chat_tab.register(demo)
        _vla_tab.register(demo)
    return demo


def main() -> None:
    """Entry point for ``python -m nanovllm_omni.serving.app``."""
    demo = build_demo()
    demo.queue().launch()


__all__ = ["build_demo", "main"]


if __name__ == "__main__":
    main()
