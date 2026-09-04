"""VLA tab -- SmolVLA action-chunk predictor (synthetic-obs path).

Single-step demo: upload one RGB observation (HWC uint8), supply a task
instruction, emit the predicted action chunk as a ``gr.Dataframe``. No
LIBERO replay -- the tab is a UI affordance for inspecting the action
shape returned by the aligned ``Omni`` seam. Heavy deps (lerobot / torch
/ numpy) live in the ``[smolvla]`` extra and are imported lazily.
"""

from __future__ import annotations

from typing import Any


def register(demo: Any) -> None:
    """Wire the SmolVLA action-chunk tab into the top-level ``gr.Blocks`` instance."""
    import gradio as gr  # noqa: PLC0415 -- lazy

    state = gr.State()

    def _ensure(state_val: Any, model_id: str) -> tuple[Any, str]:
        if state_val is not None and state_val[1] == model_id:
            return state_val
        from nanovllm_omni import Omni  # noqa: PLC0415 -- lazy by design

        # int8 matches the RTX 3050 4 GB smoke recipe (synthetic_obs.py).
        # The stage factory honors ``local_files_only=True`` by default.
        omni = Omni(model_id, dtype="int8", extra={"allow_hf_download": False})
        return (omni, model_id)

    def _predict(
        state_val: Any,
        image: Any,
        task: str,
        model_id: str,
    ) -> tuple[Any, Any]:
        if not task or not task.strip():
            raise gr.Error("Task description is required.")
        engine, mid = _ensure(state_val, model_id)
        import numpy as np  # noqa: PLC0415 -- lazy
        from PIL import Image as PILImage  # noqa: PLC0415 -- lazy

        from nanovllm_omni import SamplingParams  # noqa: PLC0415 -- lazy

        if image is None:
            raise gr.Error("Upload an RGB observation.")
        if isinstance(image, PILImage.Image):
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                # RGBA upload -- drop the alpha channel so the stage sees HWC RGB.
                arr = arr[..., :3]
        # The synthetic-obs demo passes a zero state; the LIBERO eval path
        # uses a real state vector. UI stub always sends zero.
        state_vec = np.zeros(8, dtype=np.float32)
        try:
            out = engine.generate(
                [task],
                SamplingParams(extra={"image": arr, "state": state_vec}),
            )[0]
        except Exception as exc:
            raise gr.Error(f"SmolVLA inference failed: {exc}") from exc
        if out.error:
            raise gr.Error(out.error)
        action = out.multimodal_output["actions"]
        # ActionArtifact.array is 2-D numpy [chunk_size, action_dim];
        # Dataframe renders list-of-lists cleanly across Gradio versions.
        return action.array.tolist(), (engine, mid)

    with gr.TabItem("VLA (SmolVLA)"):
        gr.Markdown(
            "**SmolVLA action-chunk predictor** -- upload one RGB observation (HWC uint8) "
            "and supply a task instruction; the stage returns the predicted action chunk "
            "as a numpy array. No LIBERO replay -- UI affordance for inspecting the action "
            "shape. Lazy-loads on first click. Heavy deps live in the `[smolvla]` extra."
        )
        model_id = gr.Textbox(label="Model id", value="HuggingFaceVLA/smolvla_libero")
        image = gr.Image(label="RGB observation (HWC uint8)", type="numpy")
        task = gr.Textbox(label="Task instruction", value="pick up the red mug")
        btn = gr.Button("Predict", variant="primary")
        out_df = gr.Dataframe(
            label="Action chunk [chunk_size, action_dim]",
            headers=None,
            datatype="number",
        )
        btn.click(
            _predict,
            inputs=[state, image, task, model_id],
            outputs=[out_df, state],
        )


__all__ = ["register"]
