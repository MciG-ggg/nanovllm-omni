from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class OmniRequestOutput:
    request_id: str = ""
    outputs: Any = None
    multimodal_output: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def from_pipeline(cls, output: Any, request_id: str = ""):
        audio = output.audio if hasattr(output, "audio") else output
        return cls(request_id=request_id, outputs=output, multimodal_output={"audio": audio})
    @classmethod
    def from_diffusion(cls, output: Any, request_id: str = ""):
        return cls(request_id=request_id, outputs=output, multimodal_output={"image": output})
    @classmethod
    def from_error(cls, error: str, request_id: str = ""):
        return cls(request_id=request_id, error=error)
    @property
    def is_pipeline_output(self): return self.multimodal_output is not None and "audio" in self.multimodal_output
    @property
    def is_diffusion_output(self): return self.multimodal_output is not None and "image" in self.multimodal_output
    def unwrap(self):
        if self.error: raise RuntimeError(self.error)
        return self.outputs
