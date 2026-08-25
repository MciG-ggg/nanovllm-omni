# Mimir-1.6B-Instruct: Offline inference

Three-stage SONAR embedding-space diffusion: SaT + SONAR encoder →
Mimir LCM (40 steps) → SONAR decoder. Output is plain text
(terminal ``final_output_type="text"``).

This family is the smallest text-output diffusion LCM in the registry
(TK-021). It runs end-to-end locally, ~8 GB VRAM required (jingrui's
RTX 5880 Ada 48 GB is comfortable; the RTX 3050 4 GB budget from
AGENTS.md is **dropped for this family** because the LCM + two SONAR
checkpoints do not fit).

## Layout

Weights follow the repo convention ``pretrained/<family_name>``. The
``run.py`` default is ``pretrained/Mimir-1.6B-Instruct`` (relative to
repo root); pass ``--model /abs/path`` to override.

```
nanovllm-omni/
└── pretrained/
    ├── Mimir-1.6B-Instruct/
    │   └── model.pt                 # mimir-lcm/Mimir-1.6B-Instruct
    ├── text_sonar_basic_encoder/
    │   └── sonar_text_encoder.pt    # dl.fbaipublicfiles.com/SONAR
    ├── text_sonar_basic_decoder/
    │   └── sonar_text_decoder.pt    # dl.fbaipublicfiles.com/SONAR
    ├── sonar_tokenizer/
    │   └── sentencepiece.source.256000.model
    ├── sat-3l/
    │   ├── model.onnx               # segment-any-text/sat-3l
    │   └── config.json
    └── xlm-roberta-base/            # SaT tokenizer dependency
        ├── tokenizer.json
        ├── config.json
        ├── tokenizer_config.json
        └── sentencepiece.bpe.model
```

Total weight footprint is ~10 GB. The model dir is symlink-friendly:
``pretrained/Mimir-1.6B-Instruct`` may be a symlink to e.g.
``/mnt/d/models/mimir-lcm/Mimir-1.6B-Instruct``.

## Setup (one-off)

```bash
# Heavy deps (sonar-space, wtpsplit, omegaconf, fairseq2) + clone lcm:
./scripts/setup_mimir.sh

# Pre-provision weights:
mkdir -p pretrained
hf download mimir-lcm/Mimir-1.6B-Instruct --local-dir pretrained/Mimir-1.6B-Instruct
curl -fL -o pretrained/text_sonar_basic_encoder/sonar_text_encoder.pt https://dl.fbaipublicfiles.com/SONAR/sonar_text_encoder.pt
curl -fL -o pretrained/text_sonar_basic_decoder/sonar_text_decoder.pt https://dl.fbaipublicfiles.com/SONAR/sonar_text_decoder.pt
curl -fL -o pretrained/sonar_tokenizer/sentencepiece.source.256000.model https://dl.fbaipublicfiles.com/SONAR/sentencepiece.source.256000.model
hf download segment-any-text/sat-3l --include model.onnx config.json --local-dir pretrained/sat-3l
hf download FacebookAI/xlm-roberta-base --include tokenizer.json config.json tokenizer_config.json sentencepiece.bpe.model --local-dir pretrained/xlm-roberta-base
```

SONAR models are not on HF — they ship from Meta's CDN
``dl.fbaipublicfiles.com/SONAR/``.

## Run

```bash
python examples/offline_inference/mimir_1_6b/run.py \
    --model pretrained/Mimir-1.6B-Instruct \
    --prompt "User turn.\n\nDefine the word 'house'.\n\nAssistant turn." \
    --output /tmp/mimir_out.txt
```

By default the run fails fast (no network) if any weight is missing.
Add ``--allow-download`` to let SONAR / SaT fetch from HF / Meta CDN
on first use. If ``sat-3l/model.onnx`` is not cached, ``prompt_encoder``
silently falls back to a regex sentence split (English-bias, fine for
short prompts).

## Pipeline

| # | stage          | kind      | input         | output         |
|---|----------------|-----------|---------------|----------------|
| 0 | prompt_encoder | CODEC     | text          | SONAR (B,S,1024) + sentences |
| 1 | mimir_lcm      | DIFFUSION | SONAR (B,S,1024) | SONAR (B,S,1024) |
| 2 | text_decoder   | CODEC     | SONAR (B,S,1024) | text (terminal) |

Heavy deps (``sonar-space``, ``wtpsplit``, ``fairseq2``, the upstream
``lcm`` clone) are imported lazily inside the stage factories, so the
``[mimir]`` extra only needs to be installed when you actually run
Mimir, not when you import ``nanovllm_omni``.
