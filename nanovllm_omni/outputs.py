import io
import wave
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

_T = TypeVar("_T")


def _is_tensor(value: Any) -> bool:
    """Detect torch tensors without making torch a base-package dependency."""
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


@dataclass(eq=False)
class MultimodalPayload(Mapping[str, Any]):
    """Mapping-compatible container for tensor outputs and metadata.

    Tensor values are kept separate from metadata, while lookup remains
    dictionary-compatible for existing vLLM-Omni consumers.
    """

    tensors: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_tensor(self) -> Any | None:
        return next(iter(self.tensors.values()), None)

    @property
    def is_empty(self) -> bool:
        return not self.tensors and not self.metadata

    def __getitem__(self, key: str) -> Any:
        if key in self.tensors:
            return self.tensors[key]
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from self.tensors
        yield from self.metadata

    def __len__(self) -> int:
        return len(self.tensors) + len(self.metadata)

    def __contains__(self, key: object) -> bool:
        return key in self.tensors or key in self.metadata

    def __bool__(self) -> bool:
        return not self.is_empty

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultimodalPayload):
            return self.tensors == other.tensors and self.metadata == other.metadata
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return NotImplemented

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.tensors)
        result.update(self.metadata)
        return result

    def merged_with(self, incoming: "MultimodalPayload") -> "MultimodalPayload":
        """Return a payload with incoming values appended/replaced by category."""
        if self.is_empty:
            return incoming
        for target, values in (
            (self.tensors, incoming.tensors),
            (self.metadata, incoming.metadata),
        ):
            for key, value in values.items():
                if key not in target:
                    target[key] = value
                elif isinstance(target[key], list):
                    target[key].append(value)
                else:
                    target[key] = [target[key], value]
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MultimodalPayload | None":
        if not data:
            return None
        tensors = {key: value for key, value in data.items() if _is_tensor(value)}
        metadata = {key: value for key, value in data.items() if not _is_tensor(value)}
        return cls(tensors=tensors, metadata=metadata)

    @classmethod
    def from_raw(cls, payload: Any, modality_key: str) -> "MultimodalPayload | None":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, Mapping):
            remapped = {
                (
                    modality_key
                    if key in {"model_outputs", "hidden"} and modality_key != "hidden"
                    else key
                ): value
                for key, value in payload.items()
            }
            return cls.from_dict(remapped)
        return cls.from_dict({modality_key: payload})


@dataclass(frozen=True)
class AudioPayload:
    data: bytes
    sample_rate: int = 24000

    def wav_bytes(self) -> bytes:
        if self.data[:4] == b"RIFF":
            return self.data
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.data)
        return out.getvalue()


@dataclass(frozen=True)
class OmniRequestOutput:
    request_id: str = ""
    outputs: Any = None
    multimodal_output: MultimodalPayload | None = None
    error: str | None = None

    @classmethod
    def from_pipeline(cls, output: Any, request_id: str = "", final_output_type: str = "audio"):
        value = output.audio if hasattr(output, "audio") else output
        payload = MultimodalPayload.from_dict({final_output_type: value})
        return cls(request_id=request_id, outputs=output, multimodal_output=payload)

    @classmethod
    def from_diffusion(cls, output: Any, request_id: str = ""):
        return cls(
            request_id=request_id,
            outputs=output,
            multimodal_output=MultimodalPayload.from_dict({"image": output}),
        )

    @classmethod
    def from_error(cls, error: str, request_id: str = ""):
        return cls(request_id=request_id, error=error)

    @property
    def is_pipeline_output(self):
        return self.multimodal_output is not None and "audio" in self.multimodal_output

    @property
    def is_diffusion_output(self):
        return self.multimodal_output is not None and "image" in self.multimodal_output

    def unwrap(self):
        if self.error:
            raise RuntimeError(self.error)
        return self.outputs


@dataclass(frozen=True)
class ActionArtifact:
    """Action output for VLA / control models. Conforms to TK-015 contract.

    ``array`` is expected to be a 2-D ``np.ndarray`` with shape
    ``[chunk_size, action_dim]``. The class does not require numpy at import
    time; the wrapper that produces an ``ActionArtifact`` owns the numpy
    import. Field validation happens in ``__post_init__`` and uses
    ``getattr`` so a torch tensor with ``.shape`` also passes.
    """

    array: Any
    action_dim: int
    chunk_size: int
    dtype: str

    @classmethod
    def from_array(cls, array: Any) -> "ActionArtifact":
        """Build from a 2-D array; auto-populate ``action_dim``/``chunk_size``/``dtype``."""
        shape = getattr(array, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(
                f"ActionArtifact.array must be 2-D [chunk_size, action_dim], got shape={shape}"
            )
        return cls(
            array=array,
            action_dim=int(shape[1]),
            chunk_size=int(shape[0]),
            dtype=str(getattr(array, "dtype", "")),
        )

    def __post_init__(self) -> None:
        shape = getattr(self.array, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(
                f"ActionArtifact.array must be 2-D [chunk_size, action_dim], got shape={shape}"
            )
        if self.action_dim != int(shape[1]):
            raise ValueError(
                f"action_dim={self.action_dim} does not match array.shape[1]={shape[1]}"
            )
        if self.chunk_size != int(shape[0]):
            raise ValueError(
                f"chunk_size={self.chunk_size} does not match array.shape[0]={shape[0]}"
            )
