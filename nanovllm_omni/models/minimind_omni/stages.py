"""MiniMind-O adapters using the existing runtime implementation."""
from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle

def create_bundle(model_id, device=None, **kwargs):
    return load_minimind_omni_bundle(model_id=model_id, device=device)

def create_stages(model_id, device=None, **kwargs):
    bundle = create_bundle(model_id, device, **kwargs)
    return bundle.thinker, bundle.talker, bundle.code2wav
