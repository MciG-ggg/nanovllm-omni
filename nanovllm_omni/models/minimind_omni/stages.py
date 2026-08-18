"""MiniMind-O stage factories and runtime bundle."""

from dataclasses import dataclass

DEFAULT_MINIMIND_MODEL_ID = "jingyaogong/minimind-3o"


@dataclass
class MinimindBundle:
    thinker: object
    talker: object
    code2wav: object


def load_minimind_omni_bundle(model_id=DEFAULT_MINIMIND_MODEL_ID, device=None, **kwargs):
    """Return a bundle placeholder; runtime integrations may provide stages."""
    return MinimindBundle(thinker=None, talker=None, code2wav=None)


def create_bundle(model_id, device=None, **kwargs):
    return load_minimind_omni_bundle(model_id=model_id, device=device, **kwargs)


def create_stages(model_id, device=None, **kwargs):
    bundle = create_bundle(model_id, device, **kwargs)
    return bundle.thinker, bundle.talker, bundle.code2wav
