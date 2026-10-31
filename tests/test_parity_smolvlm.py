"""SmolVLM greedy text parity vs HF Idefics3 (docs/adr/0011-parity-standard.md P1)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL.Image")

MODEL = os.environ.get("SMOLVLM_MODEL_DIR", "HuggingFaceTB/SmolVLM-500M-Instruct")
MAX_NEW_TOKENS = 16


def _skip_unless_ready() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if os.path.isdir(MODEL):
        return
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(f"SmolVLM checkpoint not cached locally: {MODEL}")


def _load_hf() -> object:
    cls = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(
        transformers, "AutoModelForVision2Seq", None
    )
    if cls is None:
        pytest.skip("transformers has no vision-to-text auto class")

    processor = transformers.AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    try:
        model = cls.from_pretrained(MODEL, local_files_only=True, dtype="auto")
    except TypeError:
        model = cls.from_pretrained(MODEL, local_files_only=True, torch_dtype="auto")

    return processor, model.cuda().eval()


def _hf_answer(processor: object, model: object, prompt: str, image: object) -> str:
    content = ([{"type": "image"}] if image is not None else []) + [
        {"type": "text", "text": prompt}
    ]
    chat = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
    )
    inputs = processor(
        text=chat, images=None if image is None else [image], return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    return processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _stage(dtype: str) -> object:
    from types import SimpleNamespace

    from nanovllm_omni.models.smolvlm.stage import SmolVLMStage

    args = SimpleNamespace(
        model=MODEL,
        device="cuda",
        dtype=dtype,
        gpu_memory_utilization=0.7,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        extra={"allow_hf_download": False},
    )

    return SmolVLMStage(None, args)


def _our_answer(stage: object, prompt: str, image: object) -> str:
    from nanovllm_omni import SamplingParams

    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        extra={"images": [image]} if image is not None else {},
    )

    return stage(prompt, sampling)


@pytest.mark.smoke
def test_smolvlm_greedy_matches_hf() -> None:
    _skip_unless_ready()
    from PIL import Image

    cases = [
        ("What is the capital of France?", None),
        ("Describe the image in one sentence.", Image.new("RGB", (384, 384), (200, 30, 30))),
    ]

    processor, model = _load_hf()
    dtype = str(model.dtype).replace("torch.", "")
    expected = [_hf_answer(processor, model, prompt, image) for prompt, image in cases]
    del model
    torch.cuda.empty_cache()

    stage = _stage(dtype)
    actual = [_our_answer(stage, prompt, image) for prompt, image in cases]

    assert actual == expected
