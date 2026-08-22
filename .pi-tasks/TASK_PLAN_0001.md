---
id: TASK_PLAN_0001
state: completed
phase: done
created_at: 2026-08-22T04:52:39.751Z
updated_at: 2026-08-22T05:21:44.922Z
title: 规划下如何优化
---

## task prompt

规划下如何优化

## decisions

Q1: 这次“优化”的落点在哪里——把已在 bench 验证过的 compile/int8/mimi-graph 接入真实 Omni 引擎路径让 Omni.generate/examples/audio.py/HTTP seam 实际变快,还是继续留在 nanovllm_omni.optim 内做测量与新技巧? 这决定改动落在引擎入口(OmniEngineArgs/base.py/load_minimind_omni_bundle)还是 optim 子包,以及是否新增公共 API 与契约测试。
A1: 把现有优化接到引擎路径——在 OmniEngineArgs 以 kwargs 加门控(对齐 SPEC 用户故事 6 的 enforce_eager 模式),构建 bundle 时按门控应用 torch.compile/int8/mimi-graph,bench 保持原样仅作测量

## loop events

- 2026-08-22T04:53:22.333Z  plan-question  strike 1/3  read({"path":"/Users/mcig/Projects/nanovllm-omni/nanovllm_omni/__init__.py"}) ×8 in last 0 calls  → restarted with hint

## handoff

handoff_at: 2026-08-22T05:21:44.919Z
decisions: 1
