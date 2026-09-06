"""MiniMind-Omni aligned model package exports.

Public surface is preserved from the previous single-file layout:
``MinimindBundle``, ``create_bundle``, ``generate_audio``, etc. all still
import from ``nanovllm_omni.models.minimind_omni``. The internals are now
split per stage (``bundle.py`` / ``thinker.py`` / ``talker.py`` /
``code2wav.py``).
"""

from .bundle import (
    DEFAULT_MIMI_MODEL_ID,
    DEFAULT_MINIMIND_MODEL_ID,
    MIMI_SAMPLE_RATE,
    MinimindBundle,
    create_bundle,
    create_stages,
    load_minimind_omni_bundle,
)
from .code2wav import MiniMindOmniCode2Wav, decode_audio, encode_wav, load_mimi_codec
from .pipeline import MINIMIND_OMNI_PIPELINE, PIPELINE
from .stage_processors import (
    Code2WavInputPayload,
    TalkerInputPayload,
    ThinkerStageOutput,
    talker2code2wav,
    thinker2talker,
)
from .talker import MiniMindOmniTalkerForConditionalGeneration, TalkerOutput, wrap_talker
from .thinker import generate_audio, run_generate, tokenize_for_generate

__all__ = [
    "DEFAULT_MINIMIND_MODEL_ID",
    "DEFAULT_MIMI_MODEL_ID",
    "MIMI_SAMPLE_RATE",
    "MINIMIND_OMNI_PIPELINE",
    "MinimindBundle",
    "PIPELINE",
    "create_bundle",
    "create_stages",
    "decode_audio",
    "encode_wav",
    "load_mimi_codec",
    "MiniMindOmniCode2Wav",
    "generate_audio",
    "load_minimind_omni_bundle",
    "MiniMindOmniTalkerForConditionalGeneration",
    "TalkerOutput",
    "ThinkerStageOutput",
    "TalkerInputPayload",
    "Code2WavInputPayload",
    "thinker2talker",
    "talker2code2wav",
    "wrap_talker",
    "run_generate",
    "tokenize_for_generate",
]
