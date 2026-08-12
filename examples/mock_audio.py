"""Write deterministic, model-free MiniMind-Omni demo audio to mock_audio.wav."""

from pathlib import Path

from nanovllm_omni.orchestrator import Orchestrator
from nanovllm_omni.pipeline import Pipeline
from nanovllm_omni.stage import FakeCode2Wav, FakeTalker, FakeThinker

if __name__ == "__main__":
    pipeline = Pipeline((FakeThinker(), FakeTalker(), FakeCode2Wav()))
    result = Orchestrator().submit(pipeline, "Hello from the mock MiniMind-Omni pipeline.")
    output = Path("mock_audio.wav")
    output.write_bytes(result.audio.wav_bytes())
    print(f"Wrote {output} at {result.audio.sample_rate} Hz via {', '.join(result.stage_names)}")
