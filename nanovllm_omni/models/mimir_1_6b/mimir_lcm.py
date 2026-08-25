"""Mimir stage 1 — two-tower diffusion LCM (DIFFUSION).

Loads the upstream ``lcm`` package (GitHub-only — see
``scripts/setup_mimir.sh``), builds the 1.6B two-tower config, loads
``model.pt`` weights, patches the dtype bug in
``TwoTowerDiffusionLCModel.sample_initial_noise_vectors`` (the upstream
README does this), and returns a forward callable that runs the LCM
consistency sampler.

Heavy deps (``lcm``, ``fairseq2``, ``omegaconf``, ``torch``) are
imported lazily so the registry imports cleanly without them. The
``monkeypatch __import__`` test in ``tests/test_mimir_1_6b.py`` locks
the import-guard contract.

The forward emits ``{"embeddings": (1, S', 1024), "stop_reason": str}``
where ``S'`` includes the prompt prefix. Stage 2 strips the prompt
prefix (handled in ``text_decoder.process_input``) and decodes.

Sampler defaults mirror the upstream README recipe:
``inference_timesteps=40``, ``guidance_scale=1.5``,
``guidance_rescale=0.7``, ``initial_noise_scale=0.6``,
``epsilon_scaling=1.00045``, ``eos_threshold=0.9``,
``stop_on_repetition_cosine_threshold=0.9``. Anything you want to
override per-request rides in ``sampling.extra``.
"""

from __future__ import annotations

from typing import Any

_MODEL_ID = "mimir-lcm/Mimir-1.6B-Instruct"
_INFERENCE_DTYPE = "float16"


def _generator_output_to_embeddings(payload: Any, prompt: str) -> Any:
    """Bridge stage 1 -> stage 2.

    Stage 1 emits ``{"embeddings": (1, S', 1024), "stop_reason": str}``
    where ``S' = prompt_len + generated_len``. Stage 2 needs the
    generated slice only (the SONAR decoder would otherwise echo the
    prompt). ``prompt_len`` rides along in ``payload["prompt_len"]``.
    """
    embeddings = payload["embeddings"]
    prompt_len = int(payload.get("prompt_len", 0))
    if prompt_len > 0 and embeddings.shape[1] > prompt_len:
        embeddings = embeddings[:, prompt_len:, :]
    return embeddings


def _patch_sample_initial_noise_vectors(model_cls: Any) -> None:
    """Apply the dtype patch from the upstream README.

    The default ``sample_initial_noise_vectors`` returns latents in a
    dtype that does not match ``self.dtype`` after the LCM attaches, so
    the first denoise step silently upcasts/downcasts and degrades
    quality. The patch keeps the noise in the model's dtype. We attach
    it at most once per process.
    """
    if getattr(model_cls.sample_initial_noise_vectors, "_mimir_patched", False):
        return
    original = model_cls.sample_initial_noise_vectors

    def _patched(self: Any, batch_size: int) -> Any:
        latents = original(self, batch_size)
        return latents.to(dtype=self.dtype)

    _patched._mimir_patched = True  # type: ignore[attr-defined]
    model_cls.sample_initial_noise_vectors = _patched


def _mimir_lcm_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory: build LCM model + generator, return forward."""
    import lcm  # noqa: F401  -- setup_fairseq2() side effect
    import torch
    from lcm.inference.two_tower_diffusion_lcm.generator import (
        DiffusionLCMGeneratorOptions,
        TwoTowerDiffusionLCMGenerator,
    )
    from lcm.models.two_tower_diffusion_lcm.archs import two_tower_diffusion_lcm_1_6B
    from lcm.models.two_tower_diffusion_lcm.builder import (
        TwoTowerDiffusionLCModel,
        create_two_tower_diffusion_lcm_model,
    )
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    extra = dict(getattr(args, "extra", None) or {})
    allow_hf = bool(extra.get("allow_hf_download", False))
    device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = getattr(args, "dtype", None) or _INFERENCE_DTYPE
    dtype = getattr(torch, dtype_str, torch.float16)
    checkpoint_path = extra.get("checkpoint_path") or _MODEL_ID

    # setup_fairseq2() must run before importing lcm.models.* — calling
    # it twice is a no-op in the upstream package.
    lcm.setup_fairseq2()
    _patch_sample_initial_noise_vectors(TwoTowerDiffusionLCModel)

    config = two_tower_diffusion_lcm_1_6B()
    model = create_two_tower_diffusion_lcm_model(config, device=torch.device(device), dtype=dtype)

    # Weight load: try local path first; fall back to a hf_hub download
    # when ``allow_hf_download`` is set and the local path is missing.
    state_dict_src = checkpoint_path
    if not _is_local_dir(checkpoint_path):
        if allow_hf:
            from huggingface_hub import hf_hub_download

            state_dict_src = hf_hub_download(
                repo_id=_MODEL_ID, filename="model.pt", local_dir=checkpoint_path
            )
        else:
            raise FileNotFoundError(
                f"mimir_lcm: checkpoint_path {checkpoint_path!r} is not a local "
                f"directory and allow_hf_download is False; pre-provision "
                f"the weights (hf download mimir-lcm/Mimir-1.6B-Instruct) "
                f"or pass extra={{'checkpoint_path': '<local>'}}."
            )
    state_dict = torch.load(state_dict_src, map_location=device)
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device=device, dtype=dtype)

    # EOS vector: a single "End of text." embedding used by the LCM
    # generator's stop rule. Computed once at factory time.
    text_embedder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device),
    )
    eos_vec = (
        text_embedder.predict(["End of text."], source_lang="eng_Latn")
        .squeeze()
        .to(device=device, dtype=dtype)
    )

    def mimir_lcm_forward(payload: Any, sampling: Any) -> Any:
        extras = (
            dict(sampling.extra)
            if sampling is not None and getattr(sampling, "extra", None)
            else {}
        )
        options = DiffusionLCMGeneratorOptions(
            eos_threshold=float(extras.get("eos_threshold", 0.9)),
            inference_timesteps=int(extras.get("inference_timesteps", 40)),
            initial_noise_scale=float(extras.get("initial_noise_scale", 0.6)),
            guidance_scale=float(extras.get("guidance_scale", 1.5)),
            guidance_rescale=float(extras.get("guidance_rescale", 0.7)),
            epsilon_scaling=float(extras.get("epsilon_scaling", 1.00045)),
            stop_on_repetition_cosine_threshold=float(
                extras.get("stop_on_repetition_cosine_threshold", 0.9)
            ),
            seed=int(extras.get("seed", 42)),
        )
        generator = TwoTowerDiffusionLCMGenerator(model, options, eos_vec=eos_vec)
        prompt_len = payload.seqs.shape[1] if hasattr(payload, "seqs") else 0
        with torch.inference_mode():
            output = generator(payload)
        # output.hypotheses[0] is a list; we take the top-1.
        hyp_seq = output.hypotheses[0][0].seq
        # The generator concatenates the prompt prefix to its output;
        # stage 2 strips it via the bridge, but we record the length
        # here so the bridge knows where to cut.
        return {
            "embeddings": hyp_seq,
            "stop_reason": getattr(output.hypotheses[0][0], "stop_reason", "unknown"),
            "prompt_len": prompt_len,
        }

    return mimir_lcm_forward


def _is_local_dir(path: str) -> bool:
    from pathlib import Path

    p = Path(path)
    return p.is_dir() or p.is_file()


__all__ = ["_generator_output_to_embeddings", "_mimir_lcm_stage"]
