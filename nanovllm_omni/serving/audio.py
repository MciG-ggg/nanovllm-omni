"""Audio tab -- MiniMind-O TTS (thinker -> talker -> Mimi codec -> WAV).

Lazy-loads the MiniMind-O ``Omni`` engine on first click; the same engine
is reused for every subsequent submission in the session. Heavy deps
(funasr / onnxruntime / torch) live in the ``[minimind]`` extra and are
imported lazily so the Gradio UI launches without them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


def register(demo: Any) -> None:
    """Wire the audio tab into the top-level ``gr.Blocks`` instance."""
    import gradio as gr  # noqa: PLC0415 -- lazy

    state = gr.State()

    def _ensure(state_val: Any, model_id: str, mimi_id: str) -> tuple[Any, str, str]:
        """Lazy-build the Omni engine. Never reload within the session."""
        if state_val is not None and state_val[1] == model_id and state_val[2] == mimi_id:
            return state_val
        # ponytail: lazy import keeps `import nanovllm_omni.serving.app` working
        # even when the user only installed the SD-Turbo extras.
        from nanovllm_omni import Omni  # noqa: PLC0415 -- lazy by design

        omni = Omni(model_id, mimi_model_id=mimi_id)
        return (omni, model_id, mimi_id)

    def _synthesize(
        state_val: Any,
        prompt: str,
        ref_audio: str | None,
        model_id: str,
        mimi_id: str,
        max_tokens: int,
    ) -> tuple[str | None, Any]:
        if not prompt or not prompt.strip():
            raise gr.Error("Prompt is required.")
        engine, mid, mimi = _ensure(state_val, model_id, mimi_id)
        from nanovllm_omni import SamplingParams  # noqa: PLC0415 -- lazy

        sp = SamplingParams(max_tokens=int(max_tokens)) if max_tokens else None
        # The MiniMind-O audio-in path (Q4) reads ``audio`` from the prompt
        # dict and feeds it through the thinker's encoder prefill. An empty
        # value falls through to text-only synthesis.
        prompt_obj: Any = prompt
        if ref_audio:
            prompt_obj = {"prompt": prompt, "audio": ref_audio}
        out = engine.generate([prompt_obj], sp)[0]
        if out.error:
            raise gr.Error(out.error)
        audio = out.multimodal_output["audio"]
        # AudioPayload.wav_bytes() is already a 24 kHz mono WAV (RIFF header
        # baked in by the codec stage); write it to a temp file so
        # ``gr.Audio(type="filepath")`` can play it back inline.
        fd, name = tempfile.mkstemp(suffix=".wav")
        Path(name).write_bytes(audio.wav_bytes())
        # Close the fd from mkstemp so the path is owned solely by Gradio
        # (otherwise some platforms refuse to delete the file on cleanup).
        import os

        os.close(fd)
        return name, (engine, mid, mimi)

    with gr.TabItem("Audio (MiniMind-O)"):
        gr.Markdown(
            "**MiniMind-O TTS** -- thinker -> talker -> Mimi codec -> 24 kHz mono WAV. "
            "Lazy-loads on first click; the same engine is reused for every subsequent "
            "submission in the session. Heavy deps live in the `[minimind]` extra."
        )
        with gr.Row():
            model_id = gr.Textbox(label="MiniMind-O model id", value="jingyaogong/minimind-3o")
            mimi_id = gr.Textbox(label="Mimi codec id", value="kyutai/mimi")
        prompt = gr.Textbox(
            label="Text prompt",
            lines=3,
            placeholder="Say hello in Mandarin.",
        )
        ref_audio = gr.Audio(
            label="Speaker-reference audio (optional)",
            type="filepath",
            sources=["upload"],
        )
        max_tokens = gr.Slider(64, 1024, value=512, step=64, label="max_tokens")
        btn = gr.Button("Synthesize", variant="primary")
        out_audio = gr.Audio(label="Output (24 kHz mono WAV)", type="filepath")
        btn.click(
            _synthesize,
            inputs=[state, prompt, ref_audio, model_id, mimi_id, max_tokens],
            outputs=[out_audio, state],
        )


__all__ = ["register"]
