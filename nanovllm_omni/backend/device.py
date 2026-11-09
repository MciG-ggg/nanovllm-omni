"""Device resolution, shared by the backends and the family helpers.

One place decides what "the local torch device" means, so the stage and
the backend cannot disagree about it.
"""

from __future__ import annotations


def default_torch_device() -> str:
    """cuda when there is one, else mps, else cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["default_torch_device"]
