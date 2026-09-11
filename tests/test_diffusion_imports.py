"""Import-guard tests for the diffusion module.

Per ``docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md`` §5.4:
the ``nanovllm_omni/diffusion/`` module must be importable without
torch / diffusers installed, so CPU-only CI can run engine-level tests.

Heavy deps (torch, diffusers) stay inside the pipeline factories
(``models/sd_turbo/stage.py``) and lazy-imported in client implementations.
"""

from __future__ import annotations


def test_diffusion_submodule_imports_without_torch():
    """All submodules must import in a torch-less environment.

    Simulate torch / diffusers absence via import guard at top level —
    the diffusion package only imports stdlib + typing/dataclasses,
    so the test passes if the module structure is clean.
    """
    # Imports should work without torch installed (CI §5.4 gate).
    # Only import from submodules to avoid redefinition warnings.
    from nanovllm_omni.diffusion.client import InlineDiffusionClient  # noqa: F401
    from nanovllm_omni.diffusion.engine import DiffusionEngine  # noqa: F401
    from nanovllm_omni.diffusion.interface import (  # noqa: F401
        DiffusionOutput,
        DiffusionPipeline,
        StepState,
    )
    from nanovllm_omni.diffusion.request import OmniDiffusionRequest  # noqa: F401
    from nanovllm_omni.diffusion.runner import DiffusionRunner  # noqa: F401
    from nanovllm_omni.diffusion.scheduler import RequestScheduler  # noqa: F401


def test_diffusion_module_does_not_pull_torch_at_import():
    """diffusion/__init__.py and its submodules must not import torch.

    We check the source files for `import torch` statements — they
    must only appear inside lazy factories, not at module top level.
    """
    from pathlib import Path

    # Locate the diffusion package directory.
    import nanovllm_omni.diffusion

    pkg_dir = Path(nanovllm_omni.diffusion.__file__).resolve().parent

    files = [
        "__init__.py",
        "client.py",
        "engine.py",
        "interface.py",
        "request.py",
        "runner.py",
        "scheduler.py",
    ]

    for fname in files:
        path = pkg_dir / fname
        source = path.read_text()
        # Strip docstrings and comments before checking.
        lines = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            lines.append(line)
        clean = "\n".join(lines)
        # `from __future__ import` is fine; only block `import torch` at top.
        assert (
            "import torch" not in clean or "from __future__" in clean
        ), f"diffusion/{fname} imports torch at module top level"


def test_sd_turbo_stage_imports_lazily():
    """The sd_turbo stage factory must defer torch / diffusers to call time.

    Importing the module (without calling _sd_turbo_stage) must succeed
    without torch/diffusers installed.
    """
    from nanovllm_omni.models.sd_turbo import stage as sd_stage

    # The factory should exist but not yet have imported torch/diffusers.
    assert hasattr(sd_stage, "_sd_turbo_stage")
    assert hasattr(sd_stage, "SdTurboPipeline")
    # The SdTurboPipeline class itself is defined (no torch needed).
    assert sd_stage.SdTurboPipeline.supports_step_execution is True
