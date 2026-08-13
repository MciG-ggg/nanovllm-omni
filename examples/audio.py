"""Real-weight MiniMind-O (``jingyaogong/minimind-3o``) audio demo.

Submits one request through the real Thinker -> Talker -> Code2Wav
pipeline and writes the decoded waveform to ``audio.wav``. Weights are
downloaded from the Hugging Face Hub on the first execute (~1 GB), not
at import time, so the rest of the package remains model-free.

The real-weight path needs ``torch`` / ``transformers`` / ``soundfile``.
Those live in the ``[minimind]`` optional extra; ``uv sync`` installs
them by default (see ``[tool.uv] default-groups``), but pip users must
opt in explicitly:

    pip install -e .[minimind]
    python examples/audio.py
"""

import sys
from pathlib import Path

# Allow `python examples/audio.py` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm_omni.config import load_config
from nanovllm_omni.models import load_minimind_omni_bundle
from nanovllm_omni.runtime import Orchestrator, build_pipeline

if __name__ == "__main__":
    # Offline local dirs (WSL air-gap). Override with env if needed.
    model_id = Path.home() / "minimind-3o"
    mimi_id = Path.home() / "mimi"
    if not model_id.is_dir():
        model_id = "jingyaogong/minimind-3o"
    if not mimi_id.is_dir():
        mimi_id = "kyutai/mimi"

    import torch

    # Pipeline topology + deploy profile come from YAML; bundle is the
    # concrete weight carrier for this model family.
    cfg, deploy = load_config(Path(__file__).resolve().parents[1] / "configs" / "minimind_omni.yaml")

    # Honour deploy.device when it's a concrete target; otherwise prefer
    # CUDA if available, fall back to CPU on OOM (4GB laptop GPUs).
    preferred = deploy.device
    device = preferred if preferred != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        bundle = load_minimind_omni_bundle(
            model_id=str(model_id),
            mimi_model_id=str(mimi_id),
            device=device,
        )
        pipeline = build_pipeline(cfg, bundle=bundle)
        result = Orchestrator().submit(
            pipeline, "Hello from the real MiniMind-O pipeline."
        )
    except torch.cuda.OutOfMemoryError:
        if device == "cpu":
            raise
        print("CUDA OOM; retrying on CPU")
        torch.cuda.empty_cache()
        bundle = load_minimind_omni_bundle(
            model_id=str(model_id),
            mimi_model_id=str(mimi_id),
            device="cpu",
        )
        pipeline = build_pipeline(cfg, bundle=bundle)
        result = Orchestrator().submit(
            pipeline, "Hello from the real MiniMind-O pipeline."
        )
        device = "cpu"

    output = Path("audio.wav")
    output.write_bytes(result.audio.wav_bytes())
    print(
        f"Wrote {output} at {result.audio.sample_rate} Hz via "
        f"{', '.join(result.stage_names)} on {device}"
    )
