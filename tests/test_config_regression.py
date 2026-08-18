from nanovllm_omni import OmniRequestOutput, SamplingParams
from nanovllm_omni.outputs import AudioPayload


def test_sampling_params_is_frozen_and_has_vllm_fields():
    params = SamplingParams(extra={"stage": "thinker"})
    assert params.temperature == 1.0
    assert params.max_tokens == 16
    assert params.extra["stage"] == "thinker"
    try:
        params.temperature = 0.2
    except Exception:
        pass
    else:
        raise AssertionError("SamplingParams must be frozen")


def test_pipeline_output_contract():
    audio = AudioPayload(b"RIFF", sample_rate=24000)
    output = OmniRequestOutput.from_pipeline({"audio": audio}, final_output_type="audio")
    assert output.is_pipeline_output
    assert not output.is_diffusion_output
    assert output.unwrap() == {"audio": audio}

    error = OmniRequestOutput.from_error("failed")
    try:
        error.unwrap()
    except RuntimeError as exc:
        assert str(exc) == "failed"
    else:
        raise AssertionError("error output must raise")
    diffusion = OmniRequestOutput.from_diffusion({"image": b"x"})
    assert diffusion.is_diffusion_output
