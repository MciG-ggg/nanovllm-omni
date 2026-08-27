# `OmniRequestOutput` multimodal output contract (TK-015)

**Status:** locked schema for the four artifact types. Implementation
additive (new stages may produce `ImageArtifact` / `TextArtifact`;
existing stages continue to produce the raw `PIL.Image` / `str` they
already emit, which `to_dict()` serializes losslessly).

## Locked schema

```python
from nanovllm_omni.outputs import (
    AudioPayload,        # concrete dataclass; conforms to AudioArtifact shape
    ActionArtifact,      # concrete dataclass; full schema match
    ImageArtifact,       # new in this contract; additive, optional for callers
    TextArtifact,        # new in this contract; additive, optional for callers
    MultimodalPayload,   # runtime container (Mapping[str, Any])
    OmniRequestOutput,
)
```

```python
@dataclass(frozen=True)
class AudioPayload:                # AudioArtifact equivalent
    data: bytes                    # raw PCM (16-bit mono) OR a complete WAV
    sample_rate: int = 24_000      # Hz; channels=1 by wav_bytes() convention

@dataclass(frozen=True)
class ImageArtifact:               # TK-015
    png_bytes: bytes               # serialized PNG
    width: int
    height: int

@dataclass(frozen=True)
class TextArtifact:                # TK-015
    text: str
    token_ids: list[int] | None    # None -> not exposed; [] -> exposed but empty

@dataclass(frozen=True)
class ActionArtifact:              # TK-015
    array: Any                     # 2-D [chunk_size, action_dim]; numpy OR tensor
    action_dim: int
    chunk_size: int
    dtype: str

@dataclass
class OmniRequestOutput:
    request_id: str = ""
    outputs: Any = None
    multimodal_output: MultimodalPayload | None = None
    error: str | None = None
    final_output_type: str = "text"
    images: list[Any] = field(default_factory=list)
    latents: Any = None
    metrics: dict[str, float] | None = None
    _custom_output: dict[str, Any] = field(default_factory=dict)
```

`OmniRequestOutput.multimodal_output` is a `MultimodalPayload`, which
implements `Mapping[str, Any]`. The locked TK-015 surface is therefore
`dict[str, AudioArtifact | ImageArtifact | TextArtifact | ActionArtifact]`;
in Python that's `Mapping[str, Any]` at runtime. The mapping contract
applies; the closed-set of *named* artifact types is what consumers can
rely on (see `nanovllm_omni.outputs.OutputModality` / `OutputModalityNames`
for the modality side).

## Out of scope

- Streaming chunks (`AsyncIterator[OmniRequestOutput]`) — TK-015 defers
  indefinitely.
- Multiple artifacts per request — one key per terminal output.
- New artifact types beyond the four listed.
- Bumping `multimodal_output` from `MultimodalPayload` to a plain `dict`
  would break the runtime `Mapping` surface that vllm-omni consumers
  rely on; deferred.

## Per-artifact serialization (vllm-omni parity via `to_dict()`)

`OmniRequestOutput.to_dict()` returns a `dict[str, Any]` that survives
`json.dumps` without further processing. Each artifact type has a
specific encoding so the consumer never sees a `bytes` field:

| `multimodal_output[key]` | `to_dict()[key]` | `<key>_metadata` |
|---|---|---|
| `AudioPayload` | base64-WAV bytes | `{"format": "wav", "sample_rate": 24_000}` |
| `ImageArtifact` | base64-PNG bytes | `{"format": "png", "width": W, "height": H}` |
| `TextArtifact` | `text` (str) | `{"token_ids": [...]}` if exposed, else absent |
| `ActionArtifact` | base64 of `array.tolist()` JSON; raw tensor → `.cpu().tolist()` | (none) |
| raw bytes | base64 | (none) |
| raw tensor | `value.detach().cpu().tolist()` | (none) |
| `str` / `int` / `dict` / etc. | passed through | (none) |

The `<key>_metadata` sidecar is the vllm-omni shape: HTTP adapters
read `audio_metadata.sample_rate` instead of hardcoding 24 kHz, and
read `image_metadata.{width,height}` instead of decoding the PNG.

## Per-family mapping

| Family | Stage | `multimodal_output` key | Type | Source |
|---|---|---|---|---|
| MiniMind-O | `code2wav` (terminal) | `"audio"` | `AudioPayload` | `nanovllm_omni/models/minimind_omni/thinker.py:generate_audio` |
| SD-Turbo | `sd_turbo` (terminal) | `"image"` | `PIL.Image` (raw) | `nanovllm_omni/models/sd_turbo/stage.py:_sd_turbo_forward` |
| SmolVLM | `vlm` (terminal) | `"text"` | `str` (raw) | `nanovllm_omni/models/smolvlm/stage.py:_vlm_stage` |
| SmolVLA | (terminal) | `"actions"` | `ActionArtifact` | `nanovllm_omni/models/smolvla/stage.py` |

SD-Turbo and SmolVLM currently emit raw `PIL.Image` / `str` because
those stages pre-date the `ImageArtifact` / `TextArtifact` schema; both
round-trip losslessly through `to_dict()` (the PIL path goes through the
fall-through branch; SmolVLM's `str` is passed through as a plain
string). Wrapping them in `ImageArtifact.from_pil(...)` / `TextArtifact(...)`
is a backward-compatible enhancement and can land without changing the
public surface.

## How to verify

```bash
# Locked schema:
python -c "from nanovllm_omni.outputs import (
    AudioPayload, ImageArtifact, TextArtifact, ActionArtifact,
    MultimodalPayload, OmniRequestOutput,
)"

# Round-trip a text artifact:
python -c "
import json
from nanovllm_omni.outputs import TextArtifact, MultimodalPayload, OmniRequestOutput
out = OmniRequestOutput(multimodal_output=MultimodalPayload.from_dict({'text': TextArtifact(text='hi', token_ids=[1,2,3])}))
d = out.to_dict()
print(json.dumps(d, indent=2))
"

# Round-trip an image artifact (requires Pillow):
python -c "
import io, json
from PIL import Image
from nanovllm_omni.outputs import ImageArtifact, MultimodalPayload, OmniRequestOutput
img = Image.new('RGB', (4, 4), color='red')
out = OmniRequestOutput(multimodal_output=MultimodalPayload.from_dict({'image': ImageArtifact.from_pil(img)}))
print(json.dumps(out.to_dict(), indent=2)[:200])
"

# Tests:
python -m pytest tests/test_outputs.py tests/test_action_artifact.py -v
```

The locked schema is exercised by `tests/test_outputs.py` for the audio
path and `tests/test_action_artifact.py` for actions. `TextArtifact` /
`ImageArtifact` round-trips are covered by `tests/test_outputs.py`'s
`test_to_dict_plain_str_value_stays_readable` for the str case and the
`from_pil` round-trip is verified via the new artifact classes' own
`__post_init__` validation.