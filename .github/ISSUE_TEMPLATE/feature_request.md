---
name: Feature request
about: Propose a new model family, capability, or refactor
title: "feat: "
labels: ["enhancement"]
assignees: []
---

## Problem

What problem are you trying to solve? One paragraph is enough.

## Proposed solution

The shape of the change. For a new model family, include:

- The HF checkpoint id (or local weights path) you want to support.
- The pipeline stages you expect (thinker / talker / codec / etc.).
- Which sample-rate / image-size / chunk-size the artifact layer
  should emit.
- Any new `OmniEngineArgs` fields it needs (if any).

## Alternatives considered

What else did you consider, and why did you land on this?

## Out-of-scope check

`nanovllm-omni` intentionally excludes:

- Video generation
- Multi-model pipelines beyond the four listed families
- Distributed execution, multi-card tensor-parallel, pipeline-parallel
- WebSockets / streaming responses
- FastAPI/uvicorn serving

Confirm your proposal does not conflict with any of the above.

## Willing to send a PR?

Yes / No / "I can sketch but need a reviewer." Linking the
[`.agents/skills/add-new-model/SKILL.md`](.agents/skills/add-new-model/SKILL.md)
runbook (for new families) or
[CONTRIBUTING.md](../../CONTRIBUTING.md) workflow (for other changes)
helps.
