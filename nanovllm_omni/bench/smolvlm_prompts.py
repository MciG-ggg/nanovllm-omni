"""Profile input set for smolvlm (3 inputs x 10 runs each, standard).

Text-only mode for parity with the minimind thinker bench. Three prompt
lengths isolate the prefill vs decode cost split. Greedy decoding for
deterministic per-step measurements.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmolVLMInput:
    """One smolvlm profile input (text-only mode).

    ``max_new_tokens`` / ``temperature`` are part of the profile contract.
    """

    id: str
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.0


SMOLVLM_INPUTS: tuple[SmolVLMInput, ...] = (
    # ~10 tokens; prefill cost negligible, decode-dominated.
    SmolVLMInput(
        id="smolvlm_short",
        prompt="What is the capital of France?",
    ),
    # ~25 tokens; balanced prefill + decode.
    SmolVLMInput(
        id="smolvlm_medium",
        prompt=(
            "Describe the main differences between Python and Rust for "
            "systems programming, focusing on memory management, "
            "concurrency, and runtime behavior."
        ),
    ),
    # ~80 tokens; prefill cost starts to dominate.
    SmolVLMInput(
        id="smolvlm_long",
        prompt=(
            "Write a detailed Python function that takes a list of integers "
            "and returns a new list containing only the prime numbers, "
            "preserving the original order. Include docstring, type hints, "
            "and at least three unit tests covering edge cases including "
            "empty lists, lists with negative numbers, and lists with "
            "duplicate primes. The function should be efficient and follow "
            "PEP 8."
        ),
    ),
)
