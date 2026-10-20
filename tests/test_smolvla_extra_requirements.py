"""The SmolVLA extra must ship PyAV.

Importing ``lerobot.policies`` pulls the policy factory, which pulls the
groot processors, which pull ``lerobot.datasets``, which requires PyAV.
Without the package every ``from lerobot.policies...`` import fails with
``"'av' is required but not installed"``, so the package has to be part
of the extra that the SmolVLA install uses.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_smolvla_extra_requires_av() -> None:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    requirements = config["project"]["optional-dependencies"]["smolvla"]
    names = {item.split(">")[0].split("=")[0].split("<")[0].strip() for item in requirements}

    assert "av" in names


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
