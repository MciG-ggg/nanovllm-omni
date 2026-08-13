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

from nanovllm_omni.models import load_minimind_omni_bundle
from nanovllm_omni.orchestrator import Orchestrator
from nanovllm_omni.pipeline import Pipeline

if __name__ == "__main__":
    bundle = load_minimind_omni_bundle()
    pipeline = Pipeline((bundle.thinker, bundle.talker, bundle.code2wav))
    result = Orchestrator().submit(pipeline, "Hello from the real MiniMind-O pipeline.")
    output = Path("audio.wav")
    output.write_bytes(result.audio.wav_bytes())
    print(
        f"Wrote {output} at {result.audio.sample_rate} Hz via " f"{', '.join(result.stage_names)}"
    )
