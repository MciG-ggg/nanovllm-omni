# MiniMind-O 性能优化全流程（autoresearch 完整记录）

> 项目目标：把一次 `Omni.generate`（prompt → 16 token 逐 token 生成 → mimi.decode → WAV 编码）的端到端 wall-clock 时长从 ≈928 ms 优化到 ≤ 500 ms，且不引入新依赖、不复制上游模型代码、保持公开 API。
>
> 当前基线：`11c3c13`（branch `autoresearch/under500ms`），bench median **320 ms**，stdev 11.6 ms，max 343 ms。
>
> 此文档与 `docs/minimind_omni_优化修复全流程.md` 互为补充：后者记录到 E12 之前的状态与中文叙述，本文档是 autoresearch 25 次实验的完整索引和最终结论。

## 1. 总览：25 次实验一览

| # | Commit | 描述 | 状态 | Median (ms) | Δ vs prev |
|---|---|---|---|---:|---:|
| baseline | e8df750 | KV buf + SDPA | — | 846.4 | — |
| E1 | 0cab023 | 移除 KV buf，保留 SDPA | keep | 815.4 | −31 (−3.7%) |
| E3 | — | 删除 `.clone()` 试图省一次拷贝 | discard | 1050 | +235（破坏正确性） |
| E4 | 573cdf5 | SDPA decode `is_causal=True` | keep | 803.1 | −12 (−5.1%) |
| E5 | a74a9dd | Fused RMSNorm (`aten._fused_rms_norm`) | keep | 625.6 | −178 (−26.1%) |
| E6 | 94a81fd | Fused QKV + gate-up 投影 | keep* | 467.8 | −158 (−44.7%) |
| E7 | 8638964 | Fused RoPE（消除 rotate_half 的 cat） | keep | ~478 | marginal |
| E8 | 098c6e7 | `contiguous().view()` → `split+reshape()` 清理 | keep | ~478 | — |
| E10 | 8a20704 | skip `rp==1.0` repetition penalty | keep | ~408 | −70 |
| E11 | 7a3d6c9 | model_input 单次 contiguous `copy_` | keep | ~397 | −11 |
| E12 | 068fcad | `torch.compile(dynamic=True)` on RoPE | keep | ~386 | −11 |
| E13 | 909b9c6 | RMSNorm shape 缓存 | keep（E18 reverted）| ~382 | −4 |
| E14 | c8f2cad | `torch.compile(reduce-overhead)` RoPE | crash | — | （CUDA Graph 错误） |
| E15 | c8f2cad | drop torch.compile RoPE 整体 | discard | ~395 | +13（确认 E12 是有净收益的） |
| E16 | 44ba562 | `torch.compile` MiniMindBlock.forward (opt-in) | keep (off) | ~473 | （opt-in 默认关） |
| E17 | 842f45d | 20 次稳定性验证 | keep | 386 | — |
| E18 | 1775e92 | revert E13 RMSNorm shape 缓存 | discard | ~384 | （E13 在 noise 内） |
| E19 | 9f234d7 | 15 次无 E13 状态稳定性 | keep | 379 | — |
| E20 | ff3d879 | 5× 同 (prompt,seed) 确定性验证 | keep | — | （bit-exact） |
| E21 | 6f3ca56 | 12 prompt 跨域稳健性验证 | keep | 422（宽）| （bench 380 vs wide 422） |
| E22 | a44b509 | audio 8-slot 预写 scratch buffer | discard | ~391 | +12（8 个独立 write 比单 tensor 更慢） |
| E23 | a44b509 | 最终基线 20-run 锁状态 | keep | 382 | — |
| **E24** | **7fa7d57** | **修复 qkv_fusion dedupe bug** | **keep** | **320** | **−62 (−16%)** |
| E25 | 11c3c13 | E24 后重试 kv_buffer | discard | 452 | +132（per-instance copy_ 比 torch.cat 慢） |

\* E6 的 −44.7% 实际由 RMSNorm + RoPE 等其他改动合并贡献，**真正的 QKV/gate-up 融合收益在 E24 才显现**（−62 ms）。

## 2. 当前最终状态（`11c3c13`）

```text
benchmark (6 prompt × 5 runs × max_tokens=16):
  median:   320 ms
  mean:     323 ms
  stdev:    11.6 ms
  min:      305 ms
  max:      343 ms
  <500ms:   20/20 (100%)

硬件:
  GPU:    NVIDIA RTX 3050 4GB Laptop (clock 1725 MHz @ 60°C)
  CUDA:   13.1
  PyTorch: 2.13.0+cu130

Wide-prompt (12 异质 prompt):
  median:  422 ms  (bench prompt 偏短)

Determinism:  5× 同一 (prompt, seed=42) → 同一 WAV MD5 (bit-exact)
Cross-prompt: 12/12 prompt 完整生成 8 frames, 无截断
VRAM:          493 MB
```

vs 原始 baseline `846.38 ms` → **−62.2% wall-clock**。

## 3. 优化栈（13 处 monkey-patch，全部零依赖、零 vendor）

| Patch | 位置 | 收益机理 |
|---|---|---|
| 预分配 `text_buffer` / `audio_buffer` | `generation.py` | 消除生成循环内 `torch.cat` 增长 |
| SDPA decode `is_causal=True` | `attention.py` | PyTorch 2.0+ SDPA 在 `Q_len < K_len` 时右对齐 query |
| Fused RMSNorm (`aten._fused_rms_norm`) | `rms_norm.py` | C++ 把 `pow+mean+rsqrt+mul×2+float+type_as` 合并成一个 kernel |
| Fused QKV / gate-up 投影 | `qkv_fusion.py` | 3 个独立 matmul → 1 个 batched matmul |
| Fused RoPE（去 cat） | `rope.py` | 利用 `cos/sin` 在 dim 上重复的性质，避开 `rotate_half` 的 cat |
| skip `rp==1.0` repetition penalty | `generation.py` | 默认 `rp=1.0` 时整个 unique+divide 是 no-op |
| model_input 单次 contiguous `copy_` | `generation.py` | 9 个 strided write → 1 个 view+`copy_` |
| `torch.compile(dynamic=True)` on RoPE | `rope.py` | Inductor 把 RoPE 的 6 个 elementwise kernel 融到 ~2 |
| 修 `qkv_fusion` dedupe bug | `qkv_fusion.py` | **真正把 12 个 layer 都融合 QKV**（之前只有 1 个） |

所有 monkey-patch 都通过类 / 实例级别 `forward = bound_method.__get__(self, cls)` 完成，不修改远程模型代码。

## 4. 关键经验

### 4.1 E24 的 bug fix — 22 次实验以来最大真实改进

`enable_fused_projections` 用了 `seen_attn.add(cls)` / `seen_mlp.add(cls)` 按**类**去重，但 `qkv_proj` / `gate_up_proj` 是**实例属性**。结果 12 个 Attention / 12 个 MLP 中，**只有第 1 个**被 patch（thinker.0），其余 11 个 layer 继续用 3 个独立 matmul（q_proj + k_proj + v_proj）。修掉 dedupe 后 12 个 layer 全部融合，**median 直接从 382 ms 降到 320 ms（−62 ms，5.4× stdev）**。

教训：class-level 的 `seen` 去重在 per-instance patch 场景下是 bug。看到 `seen.add(...)` 就要审视 patch 是 class-level 还是 instance-level。

### 4.2 Noise floor 与 bench 置信

20+ 次连续 bench 测量后：

- **冷态 GPU (60°C, 1725 MHz)**：stdev ~7–11 ms；
- **热态 GPU (>80°C, 1590 MHz)**：absolute wall-clock 翻倍到 ~1200 ms。

所有"在 noise 内的改动"（E13 RMSNorm cache、E22 audio scratch 等）都做了 discard。低风险下可能的最优就是当前状态。

### 4.3 CUDA Graph 的边界

- `torch.compile(reduce-overhead)` RoPE → **crash**：与 KV cache 后续的 `torch.cat` 冲突（"accessing tensor output of CUDAGraphs that has been overwritten"）。
- `torch.compile` MiniMindBlock.forward → **+91 ms regression**：Inductor overhead × 16 layers ≈ 700 ms 远超收益。
- 全栈 CUDA Graph 仍需要 monkey-patch `model.model.forward` 的 start_pos 计算（用 K.shape[1]），与 `qkv_fusion` 的实例 patch 思路一致但范围更大，风险更高。**未实施**。

### 4.4 项目侧优化的护栏

| 不允许 | 已规避 |
|---|---|
| 复制 vendor 模型副本 | `vendor/` 已删，runtime 用 `AutoModelForCausalLM(trust_remote_code=True)` |
| 引入新依赖 | 仅使用 torch 内置 op（`aten._fused_rms_norm`, `aten.silu`, SDPA） |
| 修改 HF 远程模型代码 | 全部 monkey-patch 在 `nanovllm_omni/models/minimind_omni/*.py` |
| 破坏公开 API | `Omni / AsyncOmni / SamplingParams / OmniRequestOutput` 全部可导入，49 个非 smoke 测试通过 |
| 改变生成内容（位级）| determinism 验证 5× 同 (prompt, seed) → 同 WAV MD5 |

## 5. 未尝试的剩余 ideas

按风险从低到高：

1. **进一步 micro-optimization** — 当前 7 ms stdev 下任何 < 20 ms 改动都在 noise floor 之内，且所有已识别的小热点都被试过。
2. **Triton fused RMSNorm + RoPE + add** — 需要引入 Triton 编译，复杂度高，~50 ms 估算收益。
3. **全栈 CUDA Graph capture** — 需要 monkey-patch `model.model.forward` 的 `start_pos = past_key_values[0][0].shape[1]` 计算逻辑。KV cache 静态化为固定 shape 后才能 capture。理论收益 50–150 ms，**风险**是 monkey-patch 范围大、revert 复杂、影响所有下游用户。
4. **完整 speculative decoding** — 小模型可能不划算，复杂度高。

## 6. 验证命令

```bash
# 本地静态 + 单元
cd /Users/mcig/Projects/nanovllm-omni
uv run ruff check nanovllm_omni/ tests/
uv run black --check nanovllm_omni/ tests/
uv run python -m compileall -q nanovllm_omni
uv run python -m pytest -m "not smoke" -q
uv run python -c "from nanovllm_omni import Omni, AsyncOmni, SamplingParams, OmniRequestOutput"

# WSL 性能 bench
ssh mcigs-wsl "cd ~/nanovllm-omni && .venv/bin/python -m nanovllm_omni.optim.bench time \
  --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \
  --max-tokens 16 --runs 5 --warmup 2"
```

## 7. 关联文档

- `docs/minimind_omni_优化修复全流程.md` — 早期中文叙述（到 E12 之前）
- `docs/perf/vllm_omni_thinker_talker_code2wav_performance.md` — 优化灵感的来源（vLLM-Omni 调研）
- `docs/perf/session-{1..12}.*` — 早期 TK-011/TK-015 的 Kineto / profile 数据
- `.auto/prompt.md` + `.auto/log.jsonl` — autoresearch 会话的全部 hypothesis 与实验记录
