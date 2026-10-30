"""Graph-vs-eager parity for the MiniMind-O thinker.

The thinker's CUDA graph is the stage default (ADR-0005, amendment
2026-10-30). Two things justify it: the graph is ~3.8x faster on the RTX
3050 (ADR-0007), and it changes no output. This test pins the second one.

``enforce_eager`` is fixed when the stage is built, and two thinker copies
do not fit a 4 GB card, so each cell runs in its own process. The parent
compares WAV bytes, which are a downstream function of every sampled token
and of the talker's sampling, so a single differing token shows up here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")

# Same convention as tests/test_minimind_hf_parity.py: real weights live
# outside the repo, so the path is overridable.
MODEL = os.environ.get("MINIMIND_MODEL_DIR", "/home/mcig/models/minimind-3o")
PROMPT = "你好，请用一句话介绍你自己。"
MAX_TOKENS = 16

_CHILD = f"""
import hashlib
import json
import sys

import torch

from nanovllm_omni import Omni, SamplingParams

omni = Omni(
    {MODEL!r},
    device="cuda",
    dtype="float16",
    pipeline="minimind_o",
    extra={{"allow_hf_download": False}},
    use_thinker_cuda_graph=sys.argv[1] == "graph",
)
with torch.no_grad():
    out = omni.generate([{PROMPT!r}], SamplingParams(max_tokens={MAX_TOKENS}, seed=7))[0]
wav = out.multimodal_output["audio"].wav_bytes()
print(json.dumps([hashlib.sha256(wav).hexdigest(), len(wav)]))
"""


def _run(cell: str) -> list:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, cell],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-2000:]

    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.smoke
def test_thinker_graph_matches_eager() -> None:
    eager = _run("eager")

    graph = _run("graph")

    assert graph == eager
