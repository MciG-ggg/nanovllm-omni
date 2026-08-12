"""Diffusion runtime: samplers and schedulers shared by image and action stages.

Modules here are intentionally small and dependency-light:

    sampler.py   -- flow matching + DDPM samplers (~200 LOC, no library deps)
    scheduler.py -- request-level diffusion batching (~150 LOC)
    audio_codec.py -- Mimi codec decoder for MiniMind-Omni (Phase 5)

These are NOT meant to replace diffusers; they wrap the parts we want to
demonstrate explicitly (e.g., flow matching ODE integration in nanovllm-omni
is handwritten rather than relying on diffusers' implementations, so the
notebooks can walk through the math).
"""

__all__ = []
