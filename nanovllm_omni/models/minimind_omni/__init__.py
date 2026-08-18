"""MiniMind-Omni aligned model package."""

from dataclasses import dataclass

from .pipeline import MINIMIND_OMNI_PIPELINE, PIPELINE
from .stages import create_bundle, create_stages

DEFAULT_MINIMIND_MODEL_ID = "jingyaogong/minimind-3o"


@dataclass
class MinimindBundle:
    thinker: object
    talker: object
    code2wav: object


def load_minimind_omni_bundle(model_id=DEFAULT_MINIMIND_MODEL_ID, device=None, **kwargs):
    """Return a bundle placeholder; runtime integrations may provide stages."""
    return MinimindBundle(thinker=None, talker=None, code2wav=None)


__all__ = [
    "DEFAULT_MINIMIND_MODEL_ID",
    "MinimindBundle",
    "load_minimind_omni_bundle",
    "MINIMIND_OMNI_PIPELINE",
    "PIPELINE",
    "create_bundle",
    "create_stages",
]
