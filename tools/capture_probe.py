"""Probe (report §13 reproducibility evidence):

Neutralizing two `freqs_cos[0, 0] == 0` host-read checks in
MiniMindOmni.forward unblocks CUDA Graph capture.

The checks (model_omni.py lines 258/261) force a GPU->host scalar read on
every forward call — a sync that invalidates stream capture. warmup fills
the buffers non-zero, so the precompute branch is dead code at capture
time. We string-patch the two lines to `if False:` (pure-Python, no GPU
read) and retry capture.

Monkey-patch is in-memory only; repo files untouched. Run on WSL (GPU).
"""

import inspect
import sys
import textwrap

import torch

sys.path.insert(0, "/home/mcig/nanovllm-omni")
import transformers_modules  # noqa: F401  (HF cached module, exec'd below)
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import attention as attn_mod  # noqa: F401
from nanovllm_omni.models.minimind_omni import create_bundle

bundle = create_bundle(model_id="/home/mcig/minimind-3o", mimi_model_id="/home/mcig/mimi")
tok = AutoTokenizer.from_pretrained("/home/mcig/minimind-3o", trust_remote_code=True)
ids = tok("Hello, how are you?", return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    out = bundle.model(input_ids=ids, past_key_values=None, use_cache=True)
past_kv = out.past_key_values
with torch.no_grad():
    nid = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    _ = bundle.model(input_ids=nid, past_key_values=past_kv, use_cache=True)

# --- string-level patch: neutralize both freq host-read checks ---
model = bundle.model

# The concrete class whose forward we patch: MiniMindOmni (from HF cache module).
cls = type(model)
src = inspect.getsource(cls.forward)
patched_src = src.replace(
    "if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: freq precomputed in warmup"
)
patched_src = patched_src.replace(
    "if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: freq precomputed in warmup"
)
assert "if False:" in patched_src, "patch did not apply"
print(f"patched {src.count('freqs_cos[0, 0]')} host-read checks -> if False", flush=True)

# Bind a patched copy of forward onto the class (in-memory only).

namespace = dict(cls.forward.__globals__)
namespace.pop("__name__", None)
namespace["__name__"] = cls.__module__
namespace["__qualname__"] = cls.__qualname__ + ".forward_capture_probe"
# inspect.getsource on a method returns 4-space-indented body; dedent first.
dedented_src = textwrap.dedent(patched_src)
exec(compile(dedented_src, "<capture-probe>", "exec"), namespace)
patched_forward = namespace["forward"]

# warmup on side stream
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        with torch.no_grad():
            patched_forward(model, input_ids=nid, past_key_values=past_kv, use_cache=True)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g), torch.no_grad():
        _ = patched_forward(model, input_ids=nid, past_key_values=past_kv, use_cache=True)
    print("CAPTURE OK with patched forward", flush=True)
    import time

    # eager baseline of the patched function (same code path, no graph)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.no_grad():
            patched_forward(model, input_ids=nid, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / 5 * 1000

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()
    replay_ms = (time.perf_counter() - t1) / 50 * 1000

    print(f"eager (patched fwd) median-ish: {eager_ms:.2f} ms", flush=True)
    print(f"graph replay: {replay_ms:.2f} ms", flush=True)
    print(
        f"speedup: {eager_ms / replay_ms:.2f}x  (delta {(eager_ms - replay_ms) / eager_ms * 100:+.1f}%)",
        flush=True,
    )
    print("REPLAY OK", flush=True)
except Exception as exc:
    print(f"CAPTURE FAILED: {exc}", flush=True)
    torch.cuda.synchronize()
    print("done-with-error", flush=True)
