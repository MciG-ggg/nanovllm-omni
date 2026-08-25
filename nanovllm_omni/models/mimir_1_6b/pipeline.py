"""Mimir-1.6B-Instruct pipeline topology (TK-021, multi-stage embedding-space LCM).

Three-stage declarative pipeline operating in SONAR embedding space:

- Stage 0 ``prompt_encoder`` (CODEC) — text -> SONAR sentence embeddings.
- Stage 1 ``mimir_lcm``      (DIFFUSION) — LCM diffusion over embeddings.
- Stage 2 ``text_decoder``   (CODEC) — embeddings -> text. Terminal.

``process_input`` bridges forward the SONAR tensor between stages. Each
stage factory imports its heavy deps lazily (sonar / wtpsplit / fairseq2
/ lcm) so that the registry imports cleanly without them installed; the
``monkeypatch __import__`` test in ``tests/test_mimir_1_6b.py`` locks
that contract.

The terminal stage's ``final_output_type="text"`` lands the decoded
string under ``OmniRequestOutput.multimodal_output["text"]`` via
``from_pipeline(final_output_type="text")``. SONAR / SaT / LCM are out
of band from the stdlib-only main contract -- this family opts into a
``[mimir]`` extra; see ``pyproject.toml``.
"""

from __future__ import annotations

from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_MIMIR_1_6B_FAMILY = "nanovllm_omni.models.mimir_1_6b"

MIMIR_1_6B_PIPELINE = PipelineConfig(
    name="mimir_1_6b",
    stages=(
        StageConfig(
            stage_id=0,
            name="prompt_encoder",
            kind=StageExecutionType.CODEC,
            factory=f"{_MIMIR_1_6B_FAMILY}.prompt_encoder:_prompt_encoder_stage",
            process_input=None,
            input_sources=(),
        ),
        StageConfig(
            stage_id=1,
            name="mimir_lcm",
            kind=StageExecutionType.DIFFUSION,
            factory=f"{_MIMIR_1_6B_FAMILY}.mimir_lcm:_mimir_lcm_stage",
            process_input=f"{_MIMIR_1_6B_FAMILY}.prompt_encoder:_embeddings_to_batch",
            input_sources=(0,),
        ),
        StageConfig(
            stage_id=2,
            name="text_decoder",
            kind=StageExecutionType.CODEC,
            factory=f"{_MIMIR_1_6B_FAMILY}.text_decoder:_text_decoder_stage",
            process_input=f"{_MIMIR_1_6B_FAMILY}.mimir_lcm:_generator_output_to_embeddings",
            input_sources=(1,),
            is_terminal=True,
            final_output_type="text",
        ),
    ),
    default_deploy_config_name="mimir_1_6b.yaml",
    registration_handles=(
        "mimir_1_6b",
        "mimir-lcm/Mimir-1.6B-Instruct",
        "mimir_1_6b_instruct",
    ),
    # Mimir ships only ``model.pt`` + ``README.md`` (no config.json /
    # tokenizer.json), so layers L1-L4 of ``try_infer_model_type`` miss.
    # L5 path basename match catches ``Mimir-1.6B-Instruct`` (normalized
    # ``mimir16binstruct``) against the ``mimir_1_6b_instruct`` handle.
    # L6 is unused since there's no ``config.json.architectures``.
)

PIPELINE = MIMIR_1_6B_PIPELINE

register_pipeline(MIMIR_1_6B_PIPELINE)
