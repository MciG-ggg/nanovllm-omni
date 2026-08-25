import dataclasses
from dataclasses import dataclass, field
from typing import Any, TypeAlias

OmniPromptType: TypeAlias = str | dict[str, Any]
"""One generate() prompt: text, or a dict carrying modal content.

The dict shape mirrors vllm-omni's OmniTextPrompt -- a ``prompt`` key plus
optional modal payload fields (``image``, ...). A text-only str or a dict
without modal fields behave identically.
"""


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    stop: list[str] | None = None
    seed: int | None = None
    n: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OmniEngineArgs:
    model: str | None = None
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    dtype: str | None = None
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(self, model: str | None = None, **kwargs: Any):
        self.model = model
        self.extra = dict(kwargs.pop("extra", None) or {})
        # Iterate dataclass fields instead of maintaining a parallel set:
        # adding a new typed field to the class now picks it up here automatically.
        for f in dataclasses.fields(self):
            if f.name in ("model", "extra"):
                continue
            setattr(self, f.name, kwargs.pop(f.name, f.default))
        self.extra.update(kwargs)
