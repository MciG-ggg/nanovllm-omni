import io
import wave
from dataclasses import dataclass
from typing import Any


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
    multimodal_output: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def from_pipeline(cls, output: Any, request_id: str = "", final_output_type: str = "audio"):
        audio = output.audio if hasattr(output, "audio") else output
        return cls(
            request_id=request_id, outputs=output, multimodal_output={final_output_type: audio}
        )

    @classmethod
    def from_diffusion(cls, output: Any, request_id: str = ""):
        return cls(request_id=request_id, outputs=output, multimodal_output={"image": output})

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
