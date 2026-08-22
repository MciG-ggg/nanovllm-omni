"""Vendored copy of jingyaogong/minimind-3o modeling code (TK-016 phase 3.a).

Source: https://huggingface.co/jingyaogong/minimind-3o
Files: ``model_omni.py``, ``model_minimind.py`` (verbatim, no modifications).

Why vendored: the project previously loaded these files via
``transformers.AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``,
which downloads and exec's the modeling code from the HF Hub at every cold
start. The dependency on a network round-trip and on the upstream repo
state was a deployment hazard. Vendoring pins the modeling code to this
repository so behavior is reproducible, the CUDA-graph work in TK-016
phase 3.c can patch the model locally, and the trust_remote_code escape
hatch is no longer required.

The vendored files are byte-for-byte identical to upstream at the commit
captured by audio.wav parity MD5 ``536fad2aba93d7b5df76067195dca26c``
(tests against any future upstream change). See ``NOTICE.md`` for the
upstream license / attribution.
"""

from __future__ import annotations

from . import model_minimind, model_omni

__all__ = ["model_minimind", "model_omni"]
