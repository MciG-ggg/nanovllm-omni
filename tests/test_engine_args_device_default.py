"""OmniEngineArgs device default prefers CUDA when torch sees a GPU.

Companion to the device-default fix in ``nanovllm_omni/config/params.py``.
The fix moved the device default from a class-level ``None`` to a lazy
runtime resolution (``_default_device()``) so that callers who do not
pass ``device=...`` land on a real backend instead of silently running
on CPU -- which is what bit SD-Turbo on the 4 GB card earlier.
"""

import builtins

from nanovllm_omni.config import params
from nanovllm_omni.config.params import OmniEngineArgs


def test_device_default_prefers_cuda_when_available():
    args = OmniEngineArgs()
    # Whatever the host has, the default must be one of the two valid
    # backends; never the legacy ``None``.
    assert args.device in {"cuda", "cpu"}
    try:
        import torch
    except ImportError:
        return  # no torch -> default fell back to cpu, nothing more to assert
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.device == expected


def test_device_kwarg_overrides_default():
    # Explicit ``device="cpu"`` must win over the auto-resolved default
    # -- the lazy helper only kicks in when the kwarg is absent.
    args = OmniEngineArgs(device="cpu")
    assert args.device == "cpu"


def test_default_device_helper_falls_back_without_torch(monkeypatch):
    """``_default_device`` returns ``"cpu"`` when torch import fails."""
    # Simulate torch being missing at the import boundary. The helper
    # catches ImportError and falls back; we do not need to remove an
    # already-imported torch -- instead we patch builtins.__import__ to
    # raise ImportError when the helper asks for torch.
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated no-torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert params._default_device() == "cpu"
