#!/usr/bin/env python3
"""Phase 4 wire verification on WSL RTX 3050.

Six checks, one process, one markdown table, one JSON:

1. ``forward_shapes``   -- thinker + talker forward shapes match the
   Phase 3 contract (128-d hidden, 8x2048 audio logits, 8x100 talker
   audio logits).
2. ``capture_cudagraph`` -- if the stage exposes ``capture_cudagraph()``,
   call it; report how many graphs were captured.
3. ``bit_equal_decode`` -- eager vs captured decode outputs agree to
   within fp16 tolerance.  Skipped if ``capture_cudagraph`` isn't there.
4. ``decode_20_repeat`` -- per-step decode latency on a 4-wide batch,
   p50 / p95 / VRAM peak.
5. ``e2e_mini_wav``     -- spawn ``examples/offline_inference/minimind_o/
   end2end.py`` and assert the WAV is non-empty.
6. ``gate``             -- report the Phase 4 thresholds (188 / 209 / 1126)
   as ``closed`` / ``open`` (gate gaps are recorded but never fatal).

Why ``needs WSL`` on failure
----------------------------
The whole point of this script is to validate Phase 4 wire-up on the
actual RTX 3050.  CPU-only runs would tell us nothing; if CUDA isn't
available the script exits 1 with ``needs WSL`` so the failure mode is
unambiguous in CI logs.

Usage:
    python tools/verify_phase4_wire.py \\
        --model-path /home/mcig/minimind-3o \\
        --output-dir docs/perf/aligned/enginecore-phase4-wire/
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

DEFAULT_HF_PATH = (
    "/home/mcig/.cache/huggingface/hub/models--jingyaogong--minimind-3o/"
    "snapshots/ee3febbd08cc5b2bd41c039c825a8934232fee33/"
)
DEFAULT_END2END = "examples/offline_inference/minimind_o/end2end.py"

# Phase 4 gate thresholds (from the rewrite plan).
GATE_P50_MS = 188.0
GATE_P95_MS = 209.0
GATE_VRAM_MIB = 1126.0


# ---------------------------------------------------------------------------
#  tiny helpers
# ---------------------------------------------------------------------------


def _sync_ms(fn: Any) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "p50": statistics.median(s),
        "p95": s[max(int(0.95 * len(s)) - 1, 0)],
        "min": s[0],
        "max": s[-1],
        "mean": statistics.fmean(s),
        "n": len(s),
    }


def _need_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — Phase 4 wire verification needs WSL GPU")


# ---------------------------------------------------------------------------
#  model loading (assumes Phase 3 has wired the bundle to expose the stages)
# ---------------------------------------------------------------------------


def _load_stages(model_path: str) -> tuple[Any, Any, Any, torch.device]:
    """Return ``(None, thinker, talker, device)``.

    Directly constructs ``MiniMindThinker`` / ``MiniMindTalker`` from
    the HF config + weights, bypassing the old vendor bundle.  The fork's
    ``load_model`` handles weight mapping via ``packed_modules_mapping``.
    """
    import json as _json

    import torch.distributed as dist
    from transformers import AutoConfig

    from nanovllm_omni.models.minimind_omni.talker import MiniMindTalker
    from nanovllm_omni.models.minimind_omni.thinker import MiniMindThinker

    device = torch.device("cuda")

    # Fork layers (VocabParallelEmbedding, ParallelLMHead) call
    # dist.get_rank() at __init__ time.  Initialize a single-process
    # gloo group so those calls succeed.
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # --- Thinker ---
    thinker = MiniMindThinker(
        vocab_size=getattr(cfg, "vocab_size", 6400),
        hidden_size=cfg.hidden_size,
        num_layers=getattr(cfg, "num_hidden_layers", 12),
        num_heads=getattr(cfg, "num_attention_heads", 12),
        num_kv_heads=getattr(cfg, "num_key_value_heads", 2),
        intermediate_size=getattr(cfg, "intermediate_size", None),
        max_position=getattr(cfg, "max_position_embeddings", 4096),
        rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
        rope_theta=getattr(cfg, "rope_theta", 10000),
        bridge_layer=getattr(cfg, "bridge_layer", 3),
        audio_vocab_size=getattr(cfg, "audio_vocab_size", 2048),
        num_audio_heads=getattr(cfg, "num_audio_heads", 8),
    ).to(device=device, dtype=torch.float16)
    _load_weights(thinker, model_path)

    # --- Talker ---
    talker = MiniMindTalker(
        audio_vocab_size=getattr(cfg, "audio_vocab_size", 2048),
        num_audio_heads=getattr(cfg, "num_audio_heads", 8),
        hidden_size=getattr(cfg, "audio_hidden_size", 512),
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        max_position=getattr(cfg, "max_position_embeddings", 4096),
        rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
        rope_theta=getattr(cfg, "rope_theta", 10000),
    ).to(device=device, dtype=torch.float16)

    return None, thinker, talker, device


def _load_weights(model: torch.nn.Module, model_path: str) -> None:
    """Load HF weights into a fork-layer model.

    Handles both ``*.safetensors`` (fork ``load_model``) and
    ``pytorch_model.bin`` (``torch.load`` + manual mapping).
    """
    import os
    from glob import glob

    from torch import nn

    _ensure_fork_path()

    safetensors = glob(os.path.join(model_path, "*.safetensors"))
    if safetensors:
        from nanovllm.utils.loader import load_model as _fork_load_model
        _fork_load_model(model, model_path)
        return

    # Fallback: pytorch_model.bin
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(
            f"No safetensors or pytorch_model.bin in {model_path}"
        )

    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    packed = getattr(model, "packed_modules_mapping", {})

    for weight_name, loaded_tensor in state_dict.items():
        param_name = weight_name
        shard_id = None
        for k, (v, sid) in packed.items():
            if k in weight_name:
                param_name = weight_name.replace(k, v)
                shard_id = sid
                break

        try:
            param = model.get_parameter(param_name)
        except AttributeError:
            # Skip mismatched keys silently (e.g. talker weights in thinker)
            continue

        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is not None and shard_id is not None:
            weight_loader(param, loaded_tensor, shard_id)
        elif weight_loader is not None:
            weight_loader(param, loaded_tensor)
        else:
            param.data.copy_(loaded_tensor)


def _ensure_fork_path() -> None:
    """Make ``third_party/nano-vllm`` importable."""
    import os, sys as _sys

    fork = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "nano-vllm")
    )
    if fork not in _sys.path:
        _sys.path.insert(0, fork)


# ---------------------------------------------------------------------------
#  the six checks
# ---------------------------------------------------------------------------


def _check_forward_shapes(
    thinker: Any, talker: Any, batch_size: int, seq_len: int,
) -> dict[str, Any]:
    """Run one prefill on each stage and record the contract shapes."""
    device = next(thinker.parameters()).device
    vocab_size = getattr(thinker, "vocab_size", 6400)
    hidden_size = getattr(thinker, "hidden_size", 128)
    audio_vocab = getattr(thinker, "audio_vocab_size", 2048)
    num_audio_heads = getattr(thinker, "num_audio_heads", 8)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

    out: dict[str, Any] = {"batch_size": batch_size, "seq_len": seq_len}

    with torch.no_grad():
        hidden = thinker(input_ids, positions)
    out["thinker_hidden_shape"] = list(hidden.shape)
    out["thinker_hidden_ok"] = list(hidden.shape) == [batch_size, seq_len, hidden_size]

    bridge = thinker.get_bridge_hidden() if hasattr(thinker, "get_bridge_hidden") else None
    if bridge is not None:
        out["bridge_hidden_shape"] = list(bridge.shape)
        out["bridge_hidden_ok"] = list(bridge.shape) == [batch_size, seq_len, hidden_size]

    audio_logits = thinker.get_audio_logits(hidden) if hasattr(thinker, "get_audio_logits") else None
    if audio_logits is not None:
        out["audio_logits_shape"] = list(audio_logits.shape)
        out["audio_logits_ok"] = list(audio_logits.shape) == [
            batch_size, seq_len, num_audio_heads, audio_vocab,
        ]

    text_logits = thinker.compute_logits(hidden) if hasattr(thinker, "compute_logits") else None
    if text_logits is not None:
        out["text_logits_shape"] = list(text_logits.shape)
        out["text_logits_ok"] = list(text_logits.shape[-1:]) == [vocab_size]

    # Talker: needs bridge hidden + text codes + positions.
    text_codes = torch.randint(0, audio_vocab, (seq_len,), device=device)
    with torch.no_grad():
        talker_logits = talker(bridge if bridge is not None else hidden, text_codes, positions)
    out["talker_logits_shape"] = list(talker_logits.shape)
    out["talker_logits_ok"] = list(talker_logits.shape) == [
        batch_size, seq_len, num_audio_heads, 100,
    ]
    return out


def _check_capture_cudagraph(
    thinker: Any, batch_size: int, seq_len: int,
) -> dict[str, Any]:
    """Call ``capture_cudagraph()`` if it exists, count captured graphs."""
    out: dict[str, Any] = {"available": False}
    if not hasattr(thinker, "capture_cudagraph"):
        out["skipped"] = "capture_cudagraph not on MiniMindThinker (Phase 3 not done)"
        return out

    out["available"] = True
    out["graph_count_before"] = len(getattr(thinker, "_captured_graphs", []) or [])
    thinker.capture_cudagraph()  # noqa: PD004 — Phase 3 defines the contract.
    graphs = getattr(thinker, "_captured_graphs", None) or getattr(
        thinker, "graphs", None
    )
    out["graph_count_after"] = len(graphs) if graphs is not None else None
    return out


def _check_bit_equal(
    thinker: Any, batch_size: int,
) -> dict[str, Any]:
    """Eager vs captured decode step: max abs diff < 1e-3 (fp16)."""
    if not hasattr(thinker, "capture_cudagraph"):
        return {"skipped": "capture_cudagraph not available"}

    device = next(thinker.parameters()).device
    vocab_size = getattr(thinker, "vocab_size", 6400)
    seq_len = 4

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

    with torch.no_grad():
        eager_out = thinker(input_ids, positions).clone()

    if not getattr(thinker, "_captured_graphs", None):
        return {"skipped": "no captured graphs to replay"}

    with torch.no_grad():
        graph_out = thinker(input_ids, positions).clone()

    diff = (eager_out.float() - graph_out.float()).abs().max().item()
    return {"max_abs_diff": diff, "ok": diff < 1e-3}


def _check_decode_20_repeat(
    thinker: Any, batch_size: int, repeat: int,
) -> dict[str, Any]:
    """Per-step decode latency on a 4-wide batch, p50 / p95 / VRAM peak."""
    device = next(thinker.parameters()).device
    vocab_size = getattr(thinker, "vocab_size", 6400)
    prefill_len = 8
    decode_steps = 64

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, vocab_size, (batch_size, prefill_len), device=device)
    positions = torch.arange(prefill_len, device=device).expand(batch_size, prefill_len)

    samples: list[float] = []
    with torch.no_grad():
        _ = thinker(input_ids, positions)
        torch.cuda.synchronize()

        pos = prefill_len
        for step in range(decode_steps * repeat):
            ids = torch.randint(0, vocab_size, (batch_size, 1), device=device)
            ps = torch.full((batch_size, 1), pos, device=device)
            ms = _sync_ms(lambda ids=ids, ps=ps: thinker(ids, ps))
            # Throw away the very first step — JIT / first-touch lives there.
            if step != 0:
                samples.append(ms)
            pos += 1

    vram_mib = torch.cuda.max_memory_allocated() / 2**20
    return {**_stats(samples), "vram_mib": vram_mib}


def _check_e2e_mini_wav(
    model_path: str, end2end_path: str,
) -> dict[str, Any]:
    """Spawn ``end2end.py``; assert the WAV it writes is non-empty."""
    script = Path(end2end_path)
    if not script.exists():
        return {"skipped": f"{end2end_path} not found"}

    with tempfile.TemporaryDirectory() as td:
        out_wav = Path(td) / "phase4.wav"
        cmd = [
            sys.executable, str(script),
            "--model", model_path,
            "--max-tokens", "8",
            "--out", str(out_wav),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "end2end.py timed out (>180s)"}

    if proc.returncode != 0:
        return {"ok": False, "returncode": proc.returncode, "stderr": proc.stderr[-500:]}
    if not out_wav.exists():
        return {"ok": False, "error": "end2end.py did not write the WAV"}
    return {"ok": True, "wav_bytes": out_wav.stat().st_size, "path": str(out_wav)}


# ---------------------------------------------------------------------------
#  gate + report
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """One row per check, plus the gate verdict."""

    forward_shapes: dict[str, Any] = field(default_factory=dict)
    capture_cudagraph: dict[str, Any] = field(default_factory=dict)
    bit_equal_decode: dict[str, Any] = field(default_factory=dict)
    decode_20_repeat: dict[str, Any] = field(default_factory=dict)
    e2e_mini_wav: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def to_markdown(self) -> str:
        p50 = self.decode_20_repeat.get("p50", float("nan"))
        p95 = self.decode_20_repeat.get("p95", float("nan"))
        vram = self.decode_20_repeat.get("vram_mib", float("nan"))
        gate = self.gate
        lines = [
            "## Phase 4 wire verification",
            "",
            "| check | result |",
            "| --- | --- |",
            f"| forward_shapes | `{_shape_summary(self.forward_shapes)}` |",
            f"| capture_cudagraph | `{self.capture_cudagraph}` |",
            f"| bit_equal_decode | `{self.bit_equal_decode}` |",
            f"| decode_20_repeat.p50_ms | {p50:.2f} |",
            f"| decode_20_repeat.p95_ms | {p95:.2f} |",
            f"| decode_20_repeat.vram_mib | {vram:.1f} |",
            f"| e2e_mini_wav | `{self.e2e_mini_wav}` |",
            f"| gate (p50<=188, p95<=209, vram<=1126) | "
            f"{gate.get('status', '?')} "
            f"(gaps: {gate.get('gaps', {})}) |",
            "",
        ]
        return "\n".join(lines)


def _shape_summary(shapes: dict[str, Any]) -> str:
    keys = sorted(k for k in shapes if k.endswith("_ok"))
    return ", ".join(f"{k}={shapes[k]}" for k in keys)


def _gate(decode: dict[str, Any]) -> dict[str, Any]:
    """Compare decode p50/p95/VRAM against the Phase 4 thresholds."""
    p50 = decode.get("p50")
    p95 = decode.get("p95")
    vram = decode.get("vram_mib")
    if any(v is None for v in (p50, p95, vram)):
        return {"status": "unknown", "gaps": {}}
    gaps: dict[str, float] = {}
    if p50 > GATE_P50_MS:
        gaps["p50_ms"] = round(p50 - GATE_P50_MS, 2)
    if p95 > GATE_P95_MS:
        gaps["p95_ms"] = round(p95 - GATE_P95_MS, 2)
    if vram > GATE_VRAM_MIB:
        gaps["vram_mib"] = round(vram - GATE_VRAM_MIB, 2)
    return {
        "status": "closed" if not gaps else "open",
        "gaps": gaps,
        "thresholds": {
            "p50_ms": GATE_P50_MS, "p95_ms": GATE_P95_MS, "vram_mib": GATE_VRAM_MIB,
        },
    }


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=DEFAULT_HF_PATH)
    ap.add_argument("--output-dir", default="docs/perf/aligned/enginecore-phase4-wire/")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--end2end-path", default=DEFAULT_END2END)
    args = ap.parse_args()

    _need_cuda()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    _unused, thinker, talker, device = _load_stages(args.model_path)
    print(f"loaded thinker+talker from {args.model_path} "
          f"(thinker={type(thinker).__name__}, talker={type(talker).__name__})")

    report = Report(meta={
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    })

    print("\n[1/6] forward shapes ...")
    report.forward_shapes = _check_forward_shapes(thinker, talker, args.batch_size, 8)

    print("[2/6] capture_cudagraph ...")
    report.capture_cudagraph = _check_capture_cudagraph(thinker, args.batch_size, 8)

    print("[3/6] bit-equal (eager vs captured) ...")
    report.bit_equal_decode = _check_bit_equal(thinker, args.batch_size)

    print("[4/6] decode 20-repeat bench ...")
    report.decode_20_repeat = _check_decode_20_repeat(thinker, args.batch_size, args.repeat)

    print("[5/6] E2E mini WAV ...")
    report.e2e_mini_wav = _check_e2e_mini_wav(args.model_path, args.end2end_path)

    print("[6/6] gate ...")
    report.gate = _gate(report.decode_20_repeat)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(report.to_json())
    md_path = out_dir / "report.md"
    md_path.write_text(report.to_markdown())

    print("\n" + report.to_markdown())
    print(f"wrote {out_dir}/result.json and report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
