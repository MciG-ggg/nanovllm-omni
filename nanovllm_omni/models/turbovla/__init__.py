"""TurboVLA model family wrapper (TK-011-rev).

Vendors a thin adapter around H-EmbodVis/TurboVLA's
``TurboVLAPolicy`` (Apache-2.0) that exposes the nanovllm-omni
``OmniRequestOutput`` contract with a single ``ActionArtifact`` entry.

The upstream ``turbovla`` package is **optional**; install via
``pip install -e ".[turbovla]"`` and the upstream repo
(``scripts/setup_turbovla.sh`` does both). Importing this module
without the upstream package installed raises a clear ``ImportError``
when ``TurboVLAOmni(...)`` is constructed, not at module import time.
"""

from __future__ import annotations

from .wrapper import TurboVLAConfig, TurboVLAOmni

__all__ = ["TurboVLAConfig", "TurboVLAOmni"]
