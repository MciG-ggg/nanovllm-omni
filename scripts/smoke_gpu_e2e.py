#!/usr/bin/env python3
"""End-to-end GPU smoke tests for diffusion engine + smolvla split stages.

Standalone script (no conftest.py dependency) — runs on WSL (RTX 3050 4GB)
with real weights in HF cache (offline).

Verifies:
  1. SD-Turbo: SdTurboPipeline → DiffusionEngine → valid PIL Image
  2. SD-Turbo: DiffusionEngine output vs legacy StableDiffusionPipeline
  3. SmolVLA: two-stage pipeline (vlm + action) → ActionArtifact
  4. SmolVLA: legacy single-stage → ActionArtifact (backward compat)
"""

import sys
import types


# ---------------------------------------------------------------------------
# Fix flash_attn stub issue before any import.
# diffusers checks flash_attn.__spec__ which is None in our stubs.
# ---------------------------------------------------------------------------
def _fix_flash_attn_stub():
    """Create a proper flash_attn stub with valid __spec__.
    Python 3.12+: find_spec raises ValueError if __spec__ is None.
    """
    import importlib.machinery

    for mod_name in [
        "flash_attn",
        "flash_attn.flash_attn_varlen_func",
        "flash_attn.flash_attn_with_kvcache",
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__spec__ = importlib.machinery.ModuleSpec(
                mod_name,
                loader=None,
                origin="<stub>",
            )
            sys.modules[mod_name] = m


_fix_flash_attn_stub()

import os  # noqa: E402

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

print(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

SDTURBO_MODEL = "stabilityai/sd-turbo"
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
SMOLVLA_MODEL = "HuggingFaceVLA/smolvla_libero"

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        print(f"\n--- {name} ---")
        fn()
        print("  ✅ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        failed += 1


# =========================================================================
# Test 1: SD-Turbo — real weights through DiffusionEngine
# =========================================================================
def test_sdturbo_real_weights():
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    from nanovllm_omni.diffusion.engine import DiffusionEngine
    from nanovllm_omni.diffusion.request import OmniDiffusionRequest
    from nanovllm_omni.diffusion.runner import DiffusionRunner
    from nanovllm_omni.models.sd_turbo.stage import SdTurboPipeline

    dtype = torch.float16
    device = "cuda"
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
    assert outputs[0].images is not None and len(outputs[0].images) == 1

    img = outputs[0].images[0]
    assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
    assert img.size == (512, 512), f"Expected 512x512, got {img.size}"

    arr = np.array(img)
    assert arr.std() > 5.0, f"Image appears blank: std={arr.std():.2f}"
    print(f"  Output: {img.size}, std={arr.std():.1f}")


# =========================================================================
# Test 2: SD-Turbo — DiffusionEngine vs legacy StableDiffusionPipeline
# =========================================================================
def test_sdturbo_matches_legacy():
    from diffusers import (
        AutoencoderKL,
        EulerDiscreteScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    from nanovllm_omni.diffusion.request import OmniDiffusionRequest
    from nanovllm_omni.diffusion.runner import DiffusionRunner
    from nanovllm_omni.models.sd_turbo.stage import SdTurboPipeline

    dtype = torch.float16
    device = "cuda"
    seed = 42
    prompt = "a sunset over mountains"

    # --- Legacy path ---
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

    # --- Our path ---
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

    # --- Our path (same Generator as legacy for deterministic comparison) ---
    gen_ours = torch.Generator(device=device).manual_seed(seed)

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

    # Match the scheduler init + use the same generator for latents
    our_pipeline.scheduler.set_timesteps(1)
    latent_shape = (1, our_pipeline.unet.config.in_channels, 512 // 8, 512 // 8)
    state.latents = torch.randn(
        latent_shape,
        generator=gen_ours,
        device=device,
        dtype=dtype,
    )
    state.latents = state.latents * our_pipeline.scheduler.init_noise_sigma

    noise_pred = runner.denoise_step(state, step=0, num_steps=1)
    runner.step_scheduler(state, noise_pred)
    output = runner.post_decode(state)
    img_ours = output.images[0]

    arr_legacy = np.array(img_legacy)
    arr_ours = np.array(img_ours)
    diff = np.abs(arr_legacy.astype(float) - arr_ours.astype(float)).mean()
    print(f"  Legacy vs ours: mean abs diff = {diff:.3f}")
    assert diff < 2.0, f"Image mismatch: diff = {diff:.3f}"


# =========================================================================
# Test 3: SmolVLA — two-stage pipeline (vlm + action)
# =========================================================================
def test_smolvla_split_stages():
    from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
    from nanovllm_omni.config.registry import (
        DeployConfig,
        PipelineConfig,
        StageConfig,
        StageExecutionType,
    )
    from nanovllm_omni.engine.runner import PipelineRunner
    from nanovllm_omni.outputs import ActionArtifact

    smolvla_family = "nanovllm_omni.models.smolvla"

    pipeline = PipelineConfig(
        name="smolvla_split_test",
        stages=(
            StageConfig(
                stage_id=0,
                name="vlm",
                kind=StageExecutionType.LLM_AR,
                factory=f"{smolvla_family}.vlm_stage:_vlm_stage",
                process_input=None,
                input_sources=(),
                is_terminal=False,
            ),
            StageConfig(
                stage_id=1,
                name="action",
                kind=StageExecutionType.DIFFUSION,
                factory=f"{smolvla_family}.action_stage:_action_stage",
                process_input=f"{smolvla_family}.stage_processors:vlm2action",
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
        extra={
            "allow_hf_download": False,
            # Action stage needs the lerobot checkpoint path.
            "action_model": "HuggingFaceVLA/smolvla_libero",
        },
    )

    # Fake HWC image + state (use PIL Image since vlm_stage expects it)
    from PIL import Image as PILImage

    fake_image = PILImage.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    fake_state = np.random.randn(7).astype(np.float32)

    sampling = SamplingParams(
        extra={
            "image": fake_image,
            "state": fake_state,
            "chunk_len": 10,
            "action_dim": 7,
        },
    )

    runner = PipelineRunner(pipeline, deploy, args)
    result = runner.run("pick up the red block", sampling)

    assert isinstance(result, ActionArtifact), f"Expected ActionArtifact, got {type(result)}"
    assert result.array.shape[1] == 7, f"Expected 7-DoF, got shape={result.array.shape}"
    print(
        f"  Action shape: {result.array.shape}, values range: [{result.array.min():.4f}, {result.array.max():.4f}]"
    )


# =========================================================================
# Test 4: SmolVLA — legacy single-stage (backward compat)
# =========================================================================
def test_smolvla_legacy_single_stage():
    try:
        import lerobot  # noqa: F401

        # Check if the required SmolVLM2 backbone is cached
        from transformers import AutoConfig

        AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct", local_files_only=True)
    except (ImportError, OSError) as e:
        print(f"  ⏭️  SKIPPED: {type(e).__name__}: {e}")
        return

    from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
    from nanovllm_omni.config.registry import (
        DeployConfig,
        PipelineConfig,
        StageConfig,
        StageExecutionType,
    )
    from nanovllm_omni.engine.runner import PipelineRunner
    from nanovllm_omni.outputs import ActionArtifact

    smolvla_family = "nanovllm_omni.models.smolvla"

    pipeline = PipelineConfig(
        name="smolvla_legacy_test",
        stages=(
            StageConfig(
                stage_id=0,
                name="vla",
                kind=StageExecutionType.LLM_GENERATION,
                factory=f"{smolvla_family}.stage:_vla_stage",
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

    from PIL import Image as PILImage

    fake_image = PILImage.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    fake_state = np.random.randn(7).astype(np.float32)

    sampling = SamplingParams(
        extra={
            "image": fake_image,
            "state": fake_state,
        },
    )

    runner = PipelineRunner(pipeline, deploy, args)
    result = runner.run("pick up the red block", sampling)

    assert isinstance(result, ActionArtifact), f"Expected ActionArtifact, got {type(result)}"
    assert result.array.ndim == 2, f"Expected 2-D, got ndim={result.array.ndim}"
    print(
        f"  Action shape: {result.array.shape}, values range: [{result.array.min():.4f}, {result.array.max():.4f}]"
    )


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    run_test("SD-Turbo: real weights → DiffusionEngine → PIL Image", test_sdturbo_real_weights)
    run_test(
        "SD-Turbo: DiffusionEngine vs legacy StableDiffusionPipeline", test_sdturbo_matches_legacy
    )
    run_test(
        "SmolVLA: two-stage pipeline (vlm + action) → ActionArtifact", test_smolvla_split_stages
    )
    run_test(
        "SmolVLA: legacy single-stage → ActionArtifact (backward compat)",
        test_smolvla_legacy_single_stage,
    )

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
