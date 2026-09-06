"""Image tab -- SD-Turbo text-to-image (1-step diffusion).

The current SD-Turbo stage is text-only; the ``init_image`` input is
reserved for a future img2img seam and is intentionally ignored by the
handler. SD-Turbo's adversarial distillation locks guidance_scale at 0.0
and num_inference_steps at 1 (see ``deploy/sd_turbo.yaml``); the slider
lets callers override for sanity checks, but anything but the defaults
will degrade quality.
"""

from __future__ import annotations

from typing import Any


def register(demo: Any) -> None:
    """Wire the SD-Turbo image tab into the top-level ``gr.Blocks`` instance."""
    import gradio as gr  # noqa: PLC0415 -- lazy

    state = gr.State()

    def _ensure(state_val: Any, model_id: str) -> tuple[Any, str]:
        if state_val is not None and state_val[1] == model_id:
            return state_val
        from nanovllm_omni import Omni  # noqa: PLC0415 -- lazy by design

        # Offline-first: the stage factory sets ``local_files_only=True``
        # unless ``allow_hf_download`` is opted in. Pre-provision the
        # snapshot via ``hf download`` before launching the UI.
        omni = Omni(model_id, extra={"allow_hf_download": False})
        return (omni, model_id)

    def _generate(
        state_val: Any,
        prompt: str,
        init_image: Any,
        model_id: str,
        steps: int,
        guidance: float,
    ) -> tuple[Any, Any]:
        if not prompt or not prompt.strip():
            raise gr.Error("Prompt is required.")
        engine, mid = _ensure(state_val, model_id)
        from nanovllm_omni import SamplingParams  # noqa: PLC0415 -- lazy

        sp_extra: dict[str, Any] = {"num_inference_steps": int(steps)}
        if guidance is not None:
            sp_extra["guidance_scale"] = float(guidance)
        out = engine.generate([prompt], SamplingParams(extra=sp_extra))[0]
        if out.error:
            raise gr.Error(out.error)
        # ``init_image`` is reserved for a future img2img seam; the current
        # SD-Turbo stage is text-only and ignores it. Surface a soft notice
        # via a stderr log without failing the request.
        if init_image is not None:
            print(
                "[serving.image] init_image ignored: "
                "current sd_turbo stage is text-only (img2img is reserved).",
                flush=True,
            )
        return out.multimodal_output["image"], (engine, mid)

    with gr.TabItem("Image (SD-Turbo)"):
        gr.Markdown(
            "**SD-Turbo text-to-image** -- 1-step diffusion, guidance_scale locked at 0.0. "
            "An init image is reserved for a future img2img seam (current stage is text-only). "
            "Lazy-loads on first click."
        )
        model_id = gr.Textbox(label="Model id", value="stabilityai/sd-turbo")
        prompt = gr.Textbox(
            label="Prompt",
            lines=2,
            placeholder="a cute cat, studio photo",
        )
        init_image = gr.Image(
            label="Init image (img2img -- reserved, currently ignored)",
            type="pil",
        )
        with gr.Row():
            steps = gr.Slider(1, 4, value=1, step=1, label="num_inference_steps")
            guidance = gr.Slider(0.0, 7.5, value=0.0, step=0.1, label="guidance_scale")
        btn = gr.Button("Generate", variant="primary")
        out_img = gr.Image(label="Output")
        btn.click(
            _generate,
            inputs=[state, prompt, init_image, model_id, steps, guidance],
            outputs=[out_img, state],
        )


__all__ = ["register"]
