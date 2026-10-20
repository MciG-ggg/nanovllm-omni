import dataclasses
from dataclasses import dataclass, field
from typing import Any, TypeAlias


def _default_device() -> str:
    """Return the preferred default device string for an Omni engine.

    Lazy torch import: ``config/params.py`` is imported by stages and
    entrypoints that may not have torch installed yet, so we never let a
    top-level ``import torch`` settle here. Returns ``"cpu"`` if torch
    is missing or CUDA is unavailable.
    """
    try:
        import torch  # noqa: PLC0415 -- lazy by design
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


OmniPromptType: TypeAlias = str | dict[str, Any]
"""One generate() prompt: text, or a dict carrying modal content.

The dict shape mirrors the reference ``OmniTextPrompt`` -- a ``prompt`` key plus
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
    # NOTE: ``enforce_eager`` is intentionally absent from this dataclass.
    # Each stage's fork ``Config`` owns its own ``enforce_eager`` flag
    # (set inside the stage code per ADR-0005), so an Omni-level knob
    # would either silently shadow or silently disagree with the
    # per-stage setting. A caller passing ``enforce_eager=...`` now
    # raises TypeError instead of being silently ignored.
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    dtype: str | None = None
    tensor_parallel_size: int = 1
    # The MiniMind-O pipeline path ignores this flag: the repo registers
    # its own ``minimind-o`` config class, so the hub's ``auto_map`` file
    # never runs (see ``models/minimind_omni/hf_config.py``).
    trust_remote_code: bool = True
    # Lazy default: ``_default_device()`` resolves at OmniEngineArgs() time
    # to "cuda" if torch sees a GPU, else "cpu". Keep the annotation as
    # ``str | None`` because the annotation is static and torch may not
    # be importable when this module is read.
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(self, model: str | None = None, **kwargs: Any):
        self.model = model
        self.extra = dict(kwargs.pop("extra", None) or {})
        # ``enforce_eager`` was intentionally removed: see the
        # dataclass-level note above. Surface the deletion loudly so a
        # caller still passing it sees TypeError instead of a
        # silently-ignored kwarg.
        if "enforce_eager" in kwargs:
            raise TypeError(
                "OmniEngineArgs no longer accepts 'enforce_eager' — "
                "each stage's fork Config owns its own flag "
                "(see ADR-0005). Drop the kwarg."
            )
        # Iterate dataclass fields instead of maintaining a parallel set:
        # adding a new typed field to the class now picks it up here automatically.
        for f in dataclasses.fields(self):
            if f.name in ("model", "extra"):
                continue
            default = f.default
            if f.name == "device" and "device" not in kwargs:
                # Lazy default: prefer CUDA when torch is available, fall
                # back to CPU. Computed at __init__ time so importing this
                # module does not require torch.
                default = _default_device()
            setattr(self, f.name, kwargs.pop(f.name, default))
        self.extra.update(kwargs)
