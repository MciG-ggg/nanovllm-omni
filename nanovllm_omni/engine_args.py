from dataclasses import dataclass, field
from typing import Any

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
        fields = {"enforce_eager","gpu_memory_utilization","max_num_seqs","max_num_batched_tokens","dtype","tensor_parallel_size","trust_remote_code","device"}
        for name in fields:
            setattr(self, name, kwargs.pop(name, getattr(type(self), name, None)))
        self.extra = dict(kwargs.pop("extra", {}) or {})
        self.extra.update(kwargs)
