#!/usr/bin/env python3
"""Phase 4 gate: Thinker through fork ModelRunner + Scheduler on WSL 3050.

Not a MiniMindThinker.forward() smoke. Hits the same path as nanovllm's
LLMEngine.step: Scheduler.schedule → ModelRunner.run → postprocess.

Talker is out of scope: MiniMindTalker.forward(bridge, codes, positions)
is not ModelRunner.forward(input_ids, positions).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from glob import glob
from pathlib import Path

import torch

torch._dynamo.config.suppress_errors = True

DEFAULT_HF_PATH = (
    "/home/mcig/.cache/huggingface/hub/models--jingyaogong--minimind-3o/"
    "snapshots/ee3febbd08cc5b2bd41c039c825a8934232fee33/"
)
GATE_P50_MS = 188.0
GATE_P95_MS = 209.0
GATE_VRAM_MIB = 1126.0

_FORK = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "third_party", "nano-vllm"))
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)


def _need_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — Phase 4 wire verification needs WSL GPU")


def _ensure_safetensors(model_path: str) -> None:
    """Fork load_model only reads *.safetensors. Convert bin once if needed."""
    if glob(os.path.join(model_path, "*.safetensors")):
        return
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"no safetensors or pytorch_model.bin in {model_path}")
    from safetensors.torch import save_file

    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    tensors = {}
    seen: set[int] = set()
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        ptr = v.untyped_storage().data_ptr()
        if ptr in seen:
            v = v.clone()
        else:
            seen.add(ptr)
        tensors[k] = v.contiguous()
    out = os.path.join(model_path, "model.safetensors")
    save_file(tensors, out)
    print(f"wrote {out} ({len(tensors)} tensors)")


def _thinker_only_dir(src: str) -> str:
    """HF dump is MiniMindOmni (talker/vision/audio_proj + ``model.`` prefix).

    MiniMindThinker is flat (``layers`` / ``embed_tokens``) like a Qwen3Model,
    not ``Qwen3ForCausalLM.model``. Fork ``load_model`` does not skip unknown
    keys. Strip ``model.`` and drop non-thinker tensors into a sibling dir.
    """
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    dst = os.path.join("/tmp", "minimind-thinker-only")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith((".json", ".py", ".jinja")):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    out_st = os.path.join(dst, "model.safetensors")
    keep_prefix = ("embed_tokens.", "layers.", "norm.", "lm_head.")
    tensors: dict[str, torch.Tensor] = {}
    seen: set[int] = set()
    src_st = glob(os.path.join(src, "*.safetensors"))
    if not src_st:
        raise FileNotFoundError(f"no safetensors in {src}")
    with safe_open(src_st[0], framework="pt", device="cpu") as f:
        for k in f:
            name = k[6:] if k.startswith("model.") else k
            if not name.startswith(keep_prefix):
                continue
            v = f.get_tensor(k)
            ptr = v.untyped_storage().data_ptr()
            if ptr in seen:
                v = v.clone()
            else:
                seen.add(ptr)
            tensors[name] = v.contiguous()
    save_file(tensors, out_st)
    print(f"thinker-only {out_st} ({len(tensors)} tensors)")
    return dst


def _make_config(model_path: str):
    from nanovllm.config import Config
    from nanovllm.engine.sequence import Sequence
    from transformers import AutoConfig

    orig = AutoConfig.from_pretrained

    def _trusted(*args, **kw):
        kw.setdefault("trust_remote_code", True)
        return orig(*args, **kw)

    AutoConfig.from_pretrained = _trusted  # type: ignore[method-assign]
    try:
        # Small warmup: fork warmup is min(max_num_batched_tokens, max_model_len)
        # sequences. 256-token warmup fits 4GB; 4096 default may not.
        cfg = Config(
            model=model_path,
            max_model_len=256,
            max_num_seqs=4,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.6,
            enforce_eager=False,
        )
    finally:
        AutoConfig.from_pretrained = orig  # type: ignore[method-assign]

    dt = getattr(cfg.hf_config, "dtype", None)
    if isinstance(dt, str):
        cfg.hf_config.dtype = getattr(torch, dt, torch.float16)
    elif dt is None:
        cfg.hf_config.dtype = getattr(cfg.hf_config, "torch_dtype", None) or torch.float16
    Sequence.block_size = cfg.kvcache_block_size
    return cfg


def _patch_store_kvcache() -> None:
    """Fork Triton KV store requires D=2^n. MiniMind D=4*96=384.

    ponytail: PyTorch scatter fallback. Upgrade: pad head_dim in fork kernel.
    """
    import nanovllm.layers.attention as attn

    orig = attn.store_kvcache

    def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
        n, num_heads, head_dim = key.shape
        d = num_heads * head_dim
        if d & (d - 1) == 0:
            return orig(key, value, k_cache, v_cache, slot_mapping)
        # clamp(min=0) is CUDA graph safe — no data-dependent branch.
        slots = slot_mapping.clamp(min=0)
        k_cache.view(-1, d)[slots] = key.reshape(n, d).to(k_cache.dtype)
        v_cache.view(-1, d)[slots] = value.reshape(n, d).to(v_cache.dtype)

    attn.store_kvcache = store_kvcache


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "p50": statistics.median(s),
        "p95": s[max(int(0.95 * len(s)) - 1, 0)],
        "mean": statistics.fmean(s),
        "n": len(s),
    }

    s = sorted(samples)
    return {
        "p50": statistics.median(s),
        "p95": s[max(int(0.95 * len(s)) - 1, 0)],
        "mean": statistics.fmean(s),
        "n": len(s),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_HF_PATH)
    parser.add_argument("--output-dir", default="docs/perf/aligned/enginecore-phase4-wire/")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    _need_cuda()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    _ensure_safetensors(args.model_path)
    model_path = _thinker_only_dir(args.model_path)
    cfg = _make_config(model_path)

    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams

    from nanovllm_omni.models.minimind_omni.thinker import MiniMindThinker

    _patch_store_kvcache()

    runner = ModelRunner(cfg, rank=0, event=None, model_class=MiniMindThinker)
    graphs = getattr(runner, "graphs", None) or {}
    print(f"ModelRunner up; graphs={list(graphs) if graphs else 'eager'}")

    scheduler = Scheduler(cfg)
    prompt = list(range(1, 9))
    # fork SamplingParams forbids greedy (temperature > 1e-10).
    sp = SamplingParams(temperature=1e-5, max_tokens=args.max_tokens, ignore_eos=True)
    for _ in range(args.batch_size):
        scheduler.add(Sequence(prompt, sp))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    decode_ms: list[float] = []
    n_prefill = n_decode = 0
    while not scheduler.is_finished():
        seqs, is_prefill = scheduler.schedule()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        token_ids = runner.run(seqs, is_prefill)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        scheduler.postprocess(seqs, token_ids, is_prefill)
        if is_prefill:
            n_prefill += 1
        else:
            n_decode += 1
            decode_ms.append(ms)
            if len(decode_ms) >= args.repeat:
                from nanovllm.engine.sequence import SequenceStatus

                for seq in list(scheduler.running):
                    seq.status = SequenceStatus.FINISHED
                    scheduler.block_manager.deallocate(seq)
                    scheduler.running.remove(seq)
                break

    vram_mib = torch.cuda.max_memory_allocated() / 2**20
    stats = _stats(decode_ms) if decode_ms else {"p50": float("nan"), "p95": float("nan"), "n": 0}
    p50, p95 = stats["p50"], stats["p95"]
    gaps = {
        "p50": max(0.0, p50 - GATE_P50_MS) if decode_ms else None,
        "p95": max(0.0, p95 - GATE_P95_MS) if decode_ms else None,
        "vram": max(0.0, vram_mib - GATE_VRAM_MIB),
    }
    closed = (
        bool(decode_ms) and p50 <= GATE_P50_MS and p95 <= GATE_P95_MS and vram_mib <= GATE_VRAM_MIB
    )
    report = {
        "graphs": list(graphs) if graphs else [],
        "n_prefill": n_prefill,
        "n_decode": n_decode,
        "decode": stats,
        "vram_mib": vram_mib,
        "gate": {"status": "closed" if closed else "open", "gaps": gaps},
        "meta": {
            "model_path": args.model_path,
            "batch_size": args.batch_size,
            "repeat": args.repeat,
        },
    }

    print(
        f"decode p50={p50:.1f} p95={p95:.1f} vram={vram_mib:.0f} "
        f"gate={report['gate']['status']} n={stats.get('n', 0)}"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out_dir / 'result.json'}")
    runner.exit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
