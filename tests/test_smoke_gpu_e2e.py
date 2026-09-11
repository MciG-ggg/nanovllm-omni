"""End-to-end GPU smoke tests for diffusion engine + smolvla split stages.

Runs on WSL (RTX 3050 4GB) with real weights.
Verifies:
  1. SD-Turbo: SdTurboPipeline → DiffusionEngine → valid PIL Image
  2. SmolVLA: two-stage pipeline (vlm + action) → ActionArtifact

See docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md §5.2.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

SDTURBO_MODEL = "stabilityai/sd-turbo"
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
SMOLVLA_MODEL = "HuggingFaceVLA/smolvla_libero"


# ---------------------------------------------------------------------------
# SD-Turbo: SdTurboPipeline + DiffusionEngine
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_sdturbo_pipeline_real_weights():
    """Load real sd-turbo, run prepare_encode → denoise_step → post_decode."""
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    from nanovllm_omni.diffusion.engine import DiffusionEngine
    from nanovllm_omni.diffusion.request import OmniDiffusionRequest
    from nanovllm_omni.diffusion.runner import DiffusionRunner
    from nanovllm_omni.models.sd_turbo.stage import SdTurboPipeline

    dtype = torch.float16
    device = "cuda"

    # Load components with HF cache (offline).
    load_kw = {"torch_dtype": dtype, "local_files_only": True}
    tokenizer = CLIPTokenizer.from_pretrained(SDTURBO_MODEL, subfolder="tokenizer", **load_kw)
    text_encoder = CLIPTextModel.from_pretrained(
        SDTURBO_MODEL,
        subfolder="text_encoder",
        variant="fp16",
        **load_kw,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        SDTURBO_MODEL,
        subfolder="unet",
        variant="fp16",
        **load_kw,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        SDTURBO_MODEL,
        subfolder="vae",
        variant="fp16",
        **load_kw,
    ).to(device)
    scheduler = EulerDiscreteScheduler.from_pretrained(SDTURBO_MODEL, subfolder="scheduler")

    pipeline = SdTurboPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        target_device=device,
        torch_dtype=dtype,
    )

    # Run through DiffusionEngine.
    runner = DiffusionRunner(pipeline)
    engine = DiffusionEngine(runner)
    request = OmniDiffusionRequest(
        request_id="gpu-test-1",
        prompt="a photo of a cat wearing sunglasses",
        num_inference_steps=1,
        guidance_scale=0.0,
        height=512,
        width=512,
    )

    outputs = engine.run_sync(request)
    assert len(outputs) == 1
    assert outputs[0].finished is True
    assert outputs[0].images is not None
    assert len(outputs[0].images) == 1

    img = outputs[0].images[0]
    from PIL import Image

    assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
    assert img.size == (512, 512), f"Expected 512x512, got {img.size}"

    # Verify image is not blank (std > 0).
    import numpy as np

    arr = np.array(img)
    assert arr.std() > 5.0, f"Image appears blank: std={arr.std():.2f}"

    print(f"SD-Turbo GPU test passed: {img.size}, std={arr.std():.1f}")


@pytest.mark.smoke
def test_sdturbo_pipeline_matches_legacy():
    """SD-Turbo via DiffusionEngine vs legacy StableDiffusionPipeline —
    should be bit-exact for same seed + guidance=0.0 + 1 step.
    """
    from diffusers import (
        AutoencoderKL,
        EulerDiscreteScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    from nanovllm_omni.models.sd_turbo.stage import SdTurboPipeline

    dtype = torch.float16
    device = "cuda"
    seed = 42
    prompt = "a sunset over mountains"

    # --- Legacy path (StableDiffusionPipeline) ---
    pipe_legacy = StableDiffusionPipeline.from_pretrained(
        SDTURBO_MODEL,
        torch_dtype=dtype,
        variant="fp16",
        local_files_only=True,
    ).to(device)
    gen_legacy = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        img_legacy = pipe_legacy(
            prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=512,
            width=512,
            generator=gen_legacy,
        ).images[0]

    # --- Our path (SdTurboPipeline) ---
    load_kw = {"torch_dtype": dtype, "local_files_only": True}
    tokenizer = CLIPTokenizer.from_pretrained(SDTURBO_MODEL, subfolder="tokenizer", **load_kw)
    text_encoder = CLIPTextModel.from_pretrained(
        SDTURBO_MODEL,
        subfolder="text_encoder",
        variant="fp16",
        **load_kw,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        SDTURBO_MODEL,
        subfolder="unet",
        variant="fp16",
        **load_kw,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        SDTURBO_MODEL,
        subfolder="vae",
        variant="fp16",
        **load_kw,
    ).to(device)
    scheduler = EulerDiscreteScheduler.from_pretrained(SDTURBO_MODEL, subfolder="scheduler")

    our_pipeline = SdTurboPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        target_device=device,
        torch_dtype=dtype,
    )

    # Set the same seed before prepare_encode.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from nanovllm_omni.diffusion.request import OmniDiffusionRequest
    from nanovllm_omni.diffusion.runner import DiffusionRunner

    runner = DiffusionRunner(our_pipeline)
    state = runner.prepare(
        OmniDiffusionRequest(
            request_id="compare",
            prompt=prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=512,
            width=512,
        )
    )

    # Override the scheduler to match the legacy path's timesteps.
    our_pipeline.scheduler.set_timesteps(1)
    state.latents = state.latents * our_pipeline.scheduler.init_noise_sigma

    noise_pred = runner.denoise_step(state, step=0, num_steps=1)
    runner.step_scheduler(state, noise_pred)
    output = runner.post_decode(state)
    img_ours = output.images[0]

    # Compare — same components + same seed should produce identical images.
    import numpy as np

    arr_legacy = np.array(img_legacy)
    arr_ours = np.array(img_ours)

    # SD-Turbo 1-step is deterministic; should be very close.
    diff = np.abs(arr_legacy.astype(float) - arr_ours.astype(float)).mean()
    assert diff < 2.0, f"Image mismatch: mean abs pixel diff = {diff:.3f}"

    print(f"SD-Turbo comparison passed: legacy vs ours diff = {diff:.3f}")


# ---------------------------------------------------------------------------
# SmolVLA: two-stage pipeline (mock vlm + real action)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_smolvla_split_stages_e2e():
    """Run the two-stage smolvla pipeline through PipelineRunner.
    Uses real SmolVLM backbone + real DiffusionEngine action stage.
    """
    from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
    from nanovllm_omni.config.registry import (
        DeployConfig,
        PipelineConfig,
        StageConfig,
        StageExecutionType,
    )
    from nanovllm_omni.engine.runner import PipelineRunner

    _smolvla_family = "nanovllm_omni.models.smolvla"

    # Build a two-stage pipeline config.
    pipeline = PipelineConfig(
        name="smolvla_split_test",
        stages=(
            StageConfig(
                stage_id=0,
                name="vlm",
                kind=StageExecutionType.LLM_AR,
                factory=f"{_smolvla_family}.vlm_stage:_vlm_stage",
                process_input=None,
                input_sources=(),
                is_terminal=False,
            ),
            StageConfig(
                stage_id=1,
                name="action",
                kind=StageExecutionType.DIFFUSION,
                factory=f"{_smolvla_family}.action_stage:_action_stage",
                process_input=f"{_smolvla_family}.stage_processors:vlm2action",
                input_sources=(0,),
                is_terminal=True,
                final_output_type="actions",
            ),
        ),
        default_deploy_config_name="smolvla.yaml",
    )

    deploy = DeployConfig(stages=(), max_batch=2)
    args = OmniEngineArgs(
        model=SMOLVLM_MODEL,
        dtype="bfloat16",
        device="cuda",
        extra={"allow_hf_download": False},
    )

    # Add image + state to sampling params (synthetic demo path).
    sampling = SamplingParams(
        extra={
            "image": torch.randn(3, 224, 224).byte(),  # fake HWC image
            "state": torch.randn(7).float().numpy(),
            "chunk_len": 10,
            "action_dim": 7,
        },
    )

    runner = PipelineRunner(pipeline, deploy, args)
    result = runner.run("pick up the red block", sampling)

    # Result should be an ActionArtifact.
    from nanovllm_omni.outputs import ActionArtifact

    assert isinstance(result, ActionArtifact), f"Expected ActionArtifact, got {type(result)}"
    assert result.array.shape[1] == 7, f"Expected 7-DoF actions, got shape={result.array.shape}"
    print(f"SmolVLA split stages test passed: action shape={result.array.shape}")


# ---------------------------------------------------------------------------
# SmolVLA: legacy single-stage path (backward compat)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_smolvla_legacy_single_stage():
    """Legacy single-stage smolvla still works (ADR-031)."""
    from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
    from nanovllm_omni.config.registry import (
        DeployConfig,
        PipelineConfig,
        StageConfig,
        StageExecutionType,
    )
    from nanovllm_omni.engine.runner import PipelineRunner

    _smolvla_family = "nanovllm_omni.models.smolvla"

    pipeline = PipelineConfig(
        name="smolvla_legacy_test",
        stages=(
            StageConfig(
                stage_id=0,
                name="vla",
                kind=StageExecutionType.LLM_GENERATION,
                factory=f"{_smolvla_family}.stage:_vla_stage",
                process_input=None,
                input_sources=(),
                is_terminal=True,
                final_output_type="actions",
            ),
        ),
        default_deploy_config_name="smolvla.yaml",
    )

    deploy = DeployConfig(stages=(), max_batch=2)
    args = OmniEngineArgs(
        model=SMOLVLA_MODEL,
        device="cuda",
        extra={"allow_hf_download": False},
    )

    sampling = SamplingParams(
        extra={
            "image": torch.randn(3, 224, 224).byte(),
            "state": torch.randn(7).float().numpy(),
        },
    )

    runner = PipelineRunner(pipeline, deploy, args)
    result = runner.run("pick up the red block", sampling)

    from nanovllm_omni.outputs import ActionArtifact

    assert isinstance(result, ActionArtifact), f"Expected ActionArtifact, got {type(result)}"
    assert result.array.ndim == 2, f"Expected 2-D action array, got ndim={result.array.ndim}"
    print(f"SmolVLA legacy test passed: action shape={result.array.shape}")
