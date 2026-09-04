"""Regression test: `enable_fused_projections` must fuse *every* Attention /
FeedForward instance, not just the first per class.

This locks the E24 fix (docs/perf/minimind-omni-under-500ms.md §4.1), the
largest single measured win in the 25-round optimization (-62 ms). The bug:
a class-level `seen_attn`/`seen_mlp` dedupe only patched the first instance
of each class, leaving 11 of 12 layers un-fused. The fix iterates
`model.modules()` and fuses per instance with no class-level skip.

Any future revert to a "seen"-style dedupe will fail here. CPU-only, no
weights, no GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanovllm_omni.models.minimind_omni.attention import enable_fused_projections


class Attention(nn.Module):
    """Minimal replica of the upstream Attention with q/k/v projections."""

    def __init__(self, feat_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_heads
        self.head_dim = feat_dim
        self.q_proj = nn.Linear(feat_dim, n_heads * feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, n_heads * feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, n_heads * feat_dim, bias=False)


class FeedForward(nn.Module):
    """Minimal replica of the upstream FeedForward with gate/up/down."""

    def __init__(self, feat_dim: int, hidden_mult: int = 2) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(feat_dim, feat_dim * hidden_mult, bias=False)
        self.up_proj = nn.Linear(feat_dim, feat_dim * hidden_mult, bias=False)
        self.down_proj = nn.Linear(feat_dim * hidden_mult, feat_dim, bias=False)


def _build_realistic_model() -> nn.Module:
    """Build a model whose module tree matches the real one: thinker.layers[i]
    and talker.layers[i], each a container with an Attention and FeedForward."""
    m = nn.Module()

    def make_stack(n: int, feat_dim: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(n):
            block = nn.ModuleDict(
                {
                    "self_attn": Attention(feat_dim, 4),
                    "feed_forward": FeedForward(feat_dim),
                }
            )
            layers.append(block)
        return layers

    m.thinker = nn.Module()
    m.thinker.layers = make_stack(8, 16)
    m.talker = nn.Module()
    m.talker.layers = make_stack(4, 16)
    return m


def _collect_fused(model: nn.Module) -> tuple[int, int, int, int]:
    """Return (total_attn, fused_attn, total_ff, fused_ff) for the model."""
    total_attn = fused_attn = total_ff = fused_ff = 0
    for mod in model.modules():
        if type(mod).__name__ == "Attention":
            total_attn += 1
            if hasattr(mod, "qkv_proj"):
                fused_attn += 1
        elif type(mod).__name__ == "FeedForward":
            total_ff += 1
            if hasattr(mod, "gate_up_proj"):
                fused_ff += 1
    return total_attn, fused_attn, total_ff, fused_ff


def test_all_attention_instances_get_qkv_fusion() -> None:
    """Every Attention instance (8 thinker + 4 talker) must get qkv_proj.

    This is the E24 regression: 12 attention layers, 0 un-fused. A
    class-level dedupe that only fuses the first would give 1 fused / 11
    un-fused.
    """
    model = _build_realistic_model()
    # Slight extension: the real model stores FeedForward as "FeedForward",
    # our stub uses module name. Just make sure classes have right names.
    # Verify pre-condition: nothing fused yet.
    ta, fa, tf, ff = _collect_fused(model)
    assert fa == 0 and ff == 0, f"precondition: nothing fused (found {fa} attn, {ff} ff fused)"

    enable_fused_projections(model)

    ta, fa, tf, ff = _collect_fused(model)
    assert ta == 12, f"expected 12 Attention instances, found {ta}"
    assert fa == 12, f"E24 regression: expected ALL 12 Attention fused, got {fa}/12"
    assert tf == 12, f"expected 12 FeedForward instances, found {tf}"
    assert ff == 12, f"E24 regression: expected ALL 12 FeedForward fused, got {ff}/12"


def test_first_and_last_layers_both_fused() -> None:
    """The bug masked when only layer 0 got fused; verify first AND last
    both fused across thinker and talker."""
    model = _build_realistic_model()
    enable_fused_projections(model)

    children = {
        "thinker_last_attn": model.thinker.layers[-1]["self_attn"],
        "talker_last_attn": model.talker.layers[-1]["self_attn"],
        "thinker_first_attn": model.thinker.layers[0]["self_attn"],
    }
    for name, attn in children.items():
        assert hasattr(
            attn, "qkv_proj"
        ), f"{name} missing qkv_proj (E24 regression: only early layers fused)"
        assert (
            hasattr(attn, "forward") and attn.forward.__name__ == "_fused_attention_forward"
        ), f"{name} forward not set to fused attention"


def test_fusion_preserves_weight_math() -> None:
    """The combined qkv weight must equal [q; k; v] concatenation, so the
    fused projection produces identical logits to 3 separate projections."""
    model = _build_realistic_model()
    attn = model.thinker.layers[0]["self_attn"]
    orig_q = attn.q_proj.weight.data.clone()
    orig_k = attn.k_proj.weight.data.clone()
    orig_v = attn.v_proj.weight.data.clone()

    enable_fused_projections(model)

    combined = attn.qkv_proj.weight.data
    head_q = attn.n_local_heads * attn.head_dim
    # q is [n_heads*head_dim, feat]; combined should be [3*n_heads*head_dim, feat]
    q_slice = combined[:head_q]
    k_slice = combined[head_q : 2 * head_q]
    v_slice = combined[2 * head_q :]
    assert torch.equal(q_slice, orig_q)
    assert torch.equal(k_slice, orig_k)
    assert torch.equal(v_slice, orig_v)

    # Gate-up fusion: torch.cat([g_w, u_w], dim=0) — gate rows first, then up.
    m = model.thinker.layers[0]["feed_forward"]
    orig_g = m.gate_proj.weight.data.clone()
    orig_u = m.up_proj.weight.data.clone()
    gu = m.gate_up_proj.weight.data
    g_rows = m.gate_proj.weight.shape[0]
    assert torch.equal(gu[:g_rows], orig_g)
    assert torch.equal(gu[g_rows:], orig_u)


def test_fusion_idempotent() -> None:
    """Running enable_fused_projections twice must not re-fuse or crash."""
    model = _build_realistic_model()
    enable_fused_projections(model)
    enable_fused_projections(model)  # second call — same behavior as bundle reload path
    _, fa, _, ff = _collect_fused(model)
    assert fa == 12 and ff == 12


if __name__ == "__main__":
    import sys

    checks = [
        test_all_attention_instances_get_qkv_fusion,
        test_first_and_last_layers_both_fused,
        test_fusion_preserves_weight_math,
        test_fusion_idempotent,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
