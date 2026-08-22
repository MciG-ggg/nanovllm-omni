---
id: TASK_AUTO_0001
state: failed
phase: done
created_at: 2026-08-22T04:25:39.785Z
updated_at: 2026-08-22T04:25:49.184Z
title: 那你做
---

## feature prompt

那你做

## clarifications

Q1: Does this machine have a CUDA GPU available and is it acceptable to download the jingyaogong/minimind-3o weights on first use, so that each ticket can actually run the model to verify its seam (Python and HTTP) is green? The spec's acceptance criteria ("all checkboxes green", wav_bytes() verified via wave.open) and its two-seam tests all require executing the real MiniMind-O pipeline end-to-end; whether verification happens locally vs. is deferred to CI decides whether each task carries a "run and verify" step, whether a mock-stage injection task (via register_pipeline()) must be added for GPU-less verification, and what "done" means for the TICKET-04 contract step.
A1: 你在ssh: mcigs-wsl上跑

## tasks

- [ ] TASK_0001  Reconcile README and docs with the code that actually exists — remove references to non-existent files and mark aspirational features as not implemented
- [ ] Introduce the vllm-omni-mirroring module structure beside the old Orchestrator (config/, engine/, entrypoints/, outputs.py, models/, deploy/, core/) with unit tests and a green Python seam proven by examples/minimind_omni.py | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): run and verify on ssh: mcigs-wsl — execute the real MiniMind-O pipeline there, downloading the jingyaogong/minimind-3o weights on first use; do not add a register_pipeline() mock-stage task
- [ ] Wire the HTTP adapter to the new engine — stdlib POST /v1/chat/completions returning OpenAI-shape chat.completion with base64 WAV in choices[0].message.audio.data | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): run and verify on ssh: mcigs-wsl — execute the real MiniMind-O pipeline there, downloading the jingyaogong/minimind-3o weights on first use; do not add a register_pipeline() mock-stage task
- [ ] Delete the old API (Orchestrator/Pipeline/Stage/Payload), rewrite examples/audio.py, and run the full verification checklist until every acceptance checkbox is green | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): run and verify on ssh: mcigs-wsl — execute the real MiniMind-O pipeline there, downloading the jingyaogong/minimind-3o weights on first use; do not add a register_pipeline() mock-stage task

