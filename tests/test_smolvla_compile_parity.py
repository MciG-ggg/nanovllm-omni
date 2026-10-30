"""How far does the compiled SmolVLA flow step drift from eager?

``--compile-flow`` cuts a request from ~1342 ms to ~437 ms (ADR-0009), but
torch.compile fuses kernels and the stage's stated contract is bit-exact
alignment with ``policy.predict_action_chunk``. So the compiled path is
off by default and this test pins what turning it on costs: the measured
drift, not exactness.

Both cells draw their initial noise from the global RNG, so the child seeds
before the run; without that the two arms are different samples and the
difference is meaningless.

One subprocess per cell: the compiled callable is built once per pipeline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")

MODEL = os.environ.get("SMOLVLA_MODEL_DIR", "/home/mcig/models/smolvla_libero")
NUM_STEPS = 10

# float32 action chunk, magnitude ~1. Kernel fusion reorders reductions, so
# exact equality is not expected. Measured drift on the RTX 3050 was 3.6e-3;
# this bound is that plus headroom, and a regression past it means the
# compiled path changed behavior rather than rounding.
DRIFT_BOUND = 1e-2

_CHILD = f"""
import json
import sys
import time

import numpy as np
import torch

from nanovllm_omni.models.smolvla.action_stage import _action_stage
from nanovllm_omni.models.smolvla.vlm_stage import _vlm_stage

compile_flow = sys.argv[1] == "compile"

class _Args:
    model = {MODEL!r}
    device = "cuda"
    extra = {{"allow_hf_download": False, "compile_flow": compile_flow}}

vlm = _vlm_stage(None, _Args())
action = _action_stage(None, _Args())

image = np.random.default_rng(0).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
sampling = type(
    "Sampling",
    (),
    {{
        "extra": {{
            "image": image,
            "state": np.zeros(8, dtype=np.float32),
            "num_inference_steps": {NUM_STEPS},
        }},
        "max_tokens": None,
        "temperature": 0.0,
        "seed": 0,
        "stop": None,
    }},
)()

from nanovllm_omni.models.smolvla.stage_processors import vlm2action

# ``prepare_encode`` draws the initial noise from the global RNG, so the two
# cells must start from the same stream or they compare two different runs.
torch.manual_seed(0)

payload = vlm2action(vlm("pick up the red mug", sampling), "pick up the red mug")
torch.cuda.synchronize()
start = time.perf_counter()
state = action.prepare(payload)
steps = state.metadata["num_steps"]
for step in range(steps):
    action.step_scheduler(state, action.denoise_step(state, step=step, num_steps=steps))
torch.cuda.synchronize()
first_ms = (time.perf_counter() - start) * 1000.0
actions = action.post_decode(state).images[0].array
print(json.dumps({{"actions": actions.tolist(), "first_ms": first_ms}}))
"""


def _run(cell: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, cell],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-2000:]

    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.smoke
def test_compiled_flow_step_drifts_within_recorded_bound() -> None:
    import numpy as np

    eager = _run("eager")

    compiled = _run("compile")

    worst = float(np.abs(np.array(eager["actions"]) - np.array(compiled["actions"])).max())

    assert worst <= DRIFT_BOUND, f"max action diff {worst:.3e} > {DRIFT_BOUND:.0e}"

    assert np.array(compiled["actions"]).shape == np.array(eager["actions"]).shape
