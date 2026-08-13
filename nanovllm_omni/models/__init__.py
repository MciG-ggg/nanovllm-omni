"""Model loaders: one file per model family.

Each loader is a thin function that returns a handle the rest of the engine
can call. We do not reimplement model forward passes here -- we delegate
to vllm (for AR), diffusers (for image diffusion), and direct safetensors
loaders (for MiniMind-Omni, which is small enough to bring up ourselves).

    ar.py        -- Qwen2.5-VL-3B via vllm                 (Phase 2)
    diffusion.py -- SD3.5-medium + Qwen-Image-Edit          (Phase 3)
    vla.py       -- InternVLA-A1                            (Phase 4)
    audio.py     -- MiniMind-Omni (thinker/talker/code2wav) (Phase 5)
"""

from nanovllm_omni.models.minimind_omni import (
    DEFAULT_MINIMIND_MODEL_ID,
    RealMiniMindBundle,
    load_minimind_omni_bundle,
)

__all__ = [
    "DEFAULT_MINIMIND_MODEL_ID",
    "RealMiniMindBundle",
    "load_minimind_omni_bundle",
]
