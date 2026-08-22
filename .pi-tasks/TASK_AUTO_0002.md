---
id: TASK_AUTO_0002
state: in_progress
phase: done
created_at: 2026-08-22T04:34:01.602Z
updated_at: 2026-08-22T04:41:36.766Z
title: 继续
---

## feature prompt

继续

## clarifications

Q1: Both aligned seams (Omni/AsyncOmni/SamplingParams/OmniRequestOutput and the stdlib /v1/chat/completions adapter) and the new config//engine//entrypoints//outputs.py/models//deploy/ structure already exist in the repo, and the only SPEC acceptance still open is the test set (test_sampling_params.py, test_engine_args.py, test_pipeline_registry.py, test_omni_request_output.py, test_pipeline_runner.py, test_pipeline_executor.py) plus the HTTP integration check — so what does "继续" target: finishing that locked acceptance, or jumping to the next deferred ticket (TICKET-05: real 3-stage MiniMind-O execution with post-EOS state machine and watchdog)? This chooses between a verify-and-backfill task list and a new-feature build list, which fork the entire breakdown.
A1: 继续 the in-flight task — finish the still-open SPEC acceptance (verify/backfill the six test files — test_sampling_params, test_engine_args, test_pipeline_registry, test_omni_request_output, test_pipeline_runner, test_pipeline_executor — plus the HTTP /v1/chat/completions integration check) before touching the deferred TICKET-05 feature build (auto-resolved — already settled by the spec)

## tasks

- [x] TASK_0002  Verify and backfill test_sampling_params.py against the accepted SamplingParams surface — confirm defaults, validation, and round-trips | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): finish the still-open acceptance before touching the deferred TICKET-05 feature build
- [ ] Verify and backfill test_engine_args.py against the accepted engine-args parsing and validation — cover required/optional fields and error paths | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): finish the still-open acceptance before touching the deferred TICKET-05 feature build
- [ ] Verify and backfill test_omni_request_output.py against the accepted OmniRequestOutput shape — cover fields, aliases, and serialization | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): finish the still-open acceptance before touching the deferred TICKET-05 feature build
- [ ] Verify and backfill test_pipeline_registry.py — assert engine-args/config wiring to registered components and unknown-key handling | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): finish the still-open acceptance before touching the deferred TICKET-05 feature build
- [ ] Verify and backfill test_pipeline_runner.py — drive a request end-to-end through the stages and assert an OmniRequestOutput with post-EOS behavior stubbed per spec | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): finish the still-open acceptance before touching the deferred TICKET-05 feature build
- [ ] Verify and backfill test_pipeline_executor.py — cover executor coordination of runner + config resolution and failure propagation | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): finish the still-open acceptance before touching the deferred TICKET-05 feature build
- [ ] Add the HTTP /v1/chat/completions integration check through the stdlib adapter — assert full request→response cycle last, wiring all backfilled layers | decisions (explicit user choices — these OVERRIDE the spec doc wherever they conflict; follow them exactly): finish the still-open acceptance before touching the deferred TICKET-05 feature build

