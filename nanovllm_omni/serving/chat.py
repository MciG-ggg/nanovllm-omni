"""Chat tab -- SmolVLM-500M-Instruct (image + text -> text).

Manual Blocks-based chat: each user turn can attach one or more images
via ``gr.MultimodalTextbox``; the engine is invoked with the latest
message + its images. The SmolVLM stage is single-turn (no multi-turn
memory); the UI history is preserved for the user's benefit but is not
fed back into the prompt. The engine is lazy-built on first send.
"""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)


def register(demo: Any) -> None:
    """Wire the SmolVLM chat tab into the top-level ``gr.Blocks`` instance."""
    import gradio as gr  # noqa: PLC0415 -- lazy

    state = gr.State()

    def _ensure(state_val: Any, model_id: str) -> tuple[Any, str]:
        if state_val is not None and state_val[1] == model_id:
            return state_val
        from nanovllm_omni import Omni  # noqa: PLC0415 -- lazy by design

        omni = Omni(model_id, extra={"allow_hf_download": False})
        return (omni, model_id)

    def _user_submit(message: dict[str, Any], history: list[dict[str, Any]]):
        """Echo the user message into the chat history; clear the textbox."""
        if not message:
            return gr.update(value=None), history or []
        text = (message.get("text") or "").strip() if isinstance(message, dict) else ""
        files = (message.get("files") or []) if isinstance(message, dict) else []
        if not text and not files:
            return gr.update(value=None), history or []
        history = list(history or [])
        if text:
            history.append({"role": "user", "content": text})
        for file_path in files:
            history.append({"role": "user", "content": {"path": str(file_path)}})
        return gr.update(value=None), history

    def _bot_reply(
        history: list[dict[str, Any]],
        state_val: Any,
        model_id: str,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Invoke the SmolVLM engine on the latest user turn; append the reply."""
        if not history:
            return history or [], state_val
        from PIL import Image as PILImage  # noqa: PLC0415 -- lazy

        from nanovllm_omni import SamplingParams  # noqa: PLC0415 -- lazy

        engine, mid = _ensure(state_val, model_id)
        text = ""
        files: list[str] = []
        recent_messages: list[dict[str, Any]] = []
        for message in reversed(history):
            if message.get("role") == "assistant":
                break
            if message.get("role") == "user":
                recent_messages.append(message)
        for message in reversed(recent_messages):
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict):
                file_data = content.get("file")
                if isinstance(file_data, dict) and file_data.get("path"):
                    files.append(str(file_data["path"]))
        images: list[Any] = []
        for f in files:
            try:
                images.append(PILImage.open(f).convert("RGB"))
            except (OSError, ValueError) as exc:
                _logger.warning("Skipping unreadable chat image %s: %s", f, exc)
        prompt = text if text else "Describe this image."
        extras: dict[str, Any] = {"images": images, "max_new_tokens": 256}
        try:
            out = engine.generate([prompt], SamplingParams(extra=extras))[0]
        except Exception as exc:  # UI boundary must return a visible error.
            _logger.exception("SmolVLM chat inference failed")
            history = list(history) + [{"role": "assistant", "content": f"[error] {exc}"}]
            return history, (engine, mid)
        reply = f"[error] {out.error}" if out.error else out.multimodal_output["text"]
        history = list(history) + [{"role": "assistant", "content": reply}]
        return history, (engine, mid)

    def _clear() -> tuple[list, None]:
        return [], None

    with gr.TabItem("Chat (SmolVLM-500M)"):
        gr.Markdown(
            "**SmolVLM-500M-Instruct** -- image + text -> text. Each submission sees only "
            "its own message + image (no multi-turn memory is fed back into the prompt). "
            "Lazy-loads on first send."
        )
        model_id = gr.Textbox(label="Model id", value="HuggingFaceTB/SmolVLM-500M-Instruct")
        chatbot = gr.Chatbot(label="Conversation", type="messages", height=420)
        msg = gr.MultimodalTextbox(
            label="Message",
            placeholder="Type a message and/or attach an image...",
            file_types=["image"],
            sources=["upload"],
        )
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
        # Two-step submission: stage the user message into history, then
        # invoke the engine and append the assistant reply. ``queue=False``
        # on the first hop so the textbox clears immediately.
        send.click(
            _user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            _bot_reply,
            inputs=[chatbot, state, model_id],
            outputs=[chatbot, state],
        )
        msg.submit(
            _user_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            _bot_reply,
            inputs=[chatbot, state, model_id],
            outputs=[chatbot, state],
        )
        clear.click(_clear, None, [chatbot, state], queue=False)


__all__ = ["register"]
