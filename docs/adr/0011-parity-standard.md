# ADR 0011: 推理一致性（parity）标准（2026-10-30）

- Status: Accepted
- Date: 2026-10-30

**结论先行：** 生产路径不承诺与 HuggingFace / 预训练 kernel 逐 bit 相同。
我们承诺的是分层一致性，每一层都有可执行的闸门。

## 1. 三层

| 层 | 问题 | 判定 | 失败含义 |
|---|---|---|---|
| **P0 自洽** | 同引擎、同输入、同 seed，跑两次是否相同？ | WAV / PNG / SHA-256 相等 | 引擎不确定 |
| **P1 行为** | greedy 下 token / 文本是否与参考一致？ | token 序列或文本完全相等 | 模型或采样算错 |
| **P2 数值** | 固定输入、固定 noise，数值是否在容差内？ | fp32 `atol=1e-4`；bf16 用实测值 + margin | 实现漂移 |

P2 不是 bitwise。只有明确标注 `exact` 的条目才要求 `torch.equal`。
SD-Turbo / SmolVLA 复用参考库组件，有资格收紧到 exact；现有闸门仍是
atol / mean 级，实测后再回填。

## 2. 为什么不要求 bitwise

vLLM 的跨实现判定 `tests/models/utils.py::check_logprobs_close` 写的是
“should be similar but not necessarily equal”：token 不同时只要求落在对方
top-k logprobs 内，不比较数值。vllm-omni 的 π0.5 parity 用 LeRobot 自己的
代码做 oracle，fp32 `atol=1e-4`，bf16 `atol=5e-2`（实测 2.95e-2）。

原因是数值链条上的每一环都会改变 bit：

- checkpoint 只存权重，不存数值执行计划；预训练 kernel 与推理 kernel 不同；
- HF 自己换 `attn_implementation`（eager / sdpa / flash）就不是 bit-stable；
- serving 的 batch 形状会改变 GEMM 与 kernel 选择；
- 跨 GPU 的归约顺序不同。

逐 bit 只有在固定整条栈（设备、dtype、shape、kernel、库版本、batch 组成）时
才有定义。那等于把 serving 钉死成参考实现。

## 3. 固定参照栈

每次 P1 / P2 必须记录：

- GPU、驱动、CUDA；
- torch、transformers、diffusers、lerobot 版本；
- dtype 与 `attn_implementation`；
- tokenizer / processor revision；
- 输入契约：MiniMind 的 thinker 必须收到 chat-template 文本；`ThinkerStage`
  对原始 prompt 自动套模板，已格式化的文本原样透传。
- 输入（prompt、图像、state）与 `max_tokens`；
- seed 或显式 noise 张量。

缺一项，结果不可复现，也不能作为闸门。

## 4. 各模型族的闸门

| 族 | 闸门 | 层 | 命令 | 当前状态 |
|---|---|---|---|---|
| SD-Turbo | `tests/test_smoke_gpu_e2e.py::test_sdturbo_pipeline_matches_legacy` | P2 | `pytest -m smoke tests/test_smoke_gpu_e2e.py -k sdturbo` | 已记录：fp16，mean abs diff < 2/255；`seed` 未接 generator |
| SmolVLM | `tests/test_parity_smolvlm.py` | P1 | `pytest -m smoke tests/test_parity_smolvlm.py` | 首次 WSL 运行待跑；greedy 文本对照 HF |
| SmolVLA | `tests/test_parity_smolvla.py` | P2 | `pytest -m smoke tests/test_parity_smolvla.py` | 首次 WSL 运行待跑；`atol=1e-4` |
| MiniMind-O | `tests/test_parity_minimind.py` → `tests/test_minimind_hf_parity.py` | P1（文本）+ P2（bridge） | `PYTHONPATH=$REPO:$REPO/third_party/nano-vllm pytest -m smoke tests/test_parity_minimind.py` | **已跑通（2026-10-31）**：P1 greedy 文本 20/20 逐 token 相等；P2 bridge 47 行 cosine 1.00000000、mean relative 0.0517%。参照栈见 §4.1 |

首次运行之前，这些是待验证的闸门，不是已经通过的结论。跑完把命令、
硬件、库版本和实测数字回填到本表。

### 4.1 MiniMind-O 实测记录（2026-10-31）

参照栈：RTX 3050 4GB Laptop / 驱动 591.86 / CUDA 13.0 / torch 2.11.0+cu130 /
transformers 5.16.1 / Python 3.12.3 / fp16 / WSL。权重 `jingyaogong/minimind-3o`
（`pytorch_model.bin`，195 tensor），Mimi `kyutai/mimi`。输入：chat-template
后的 27 token prompt（`你好，请用一句话介绍你自己。`），`max_tokens=24`，
temperature=0（greedy），`rp=1.0`，seed 42。

| 层级 | 度量 | 结果 |
|---|---|---|
| P1 | greedy 文本 token | 20/20 相等，首个分歧点无 |
| P2 | thinker 文本 logits（全 27 个位置） | max abs 0.03125，mean abs 0.00259（相对 0.097%），argmax 27/27 = 100%，ours argmax 全部落在 vendor top-1 |
| P2 | bridge 行（layer 3 输出） | cosine 1.00000000，mean relative 0.05%，最大元素差 1.0（在 ~1004 量级上 = 1 个 fp16 ULP） |

结论：fork thinker（paged attention + 融合 kernel + fp16）与 vendor 的 eager
全序列实现在 fp16 舍入内等价。分歧只剩 RNG：vendor 用 `torch.multinomial`，
fork 用 Gumbel-max，取数顺序不同，**采样路径下的 code 相同不是预期**。
旧版闸门把两侧**采样后**的音频码逐位置比较（阈值 87.5%），实测 7.81%——
那不是实现错误，是 RNG 交错。闸门已改成上面的 P1/P2，采样码只做信息输出。

这些测试都带 `@pytest.mark.smoke`，CI 的 `-m "not smoke"` 不跑它们。
它们在 WSL / RTX 3050 上手动执行，结果写回对应文档。

## 5. 已知缺口（闸门尚未覆盖）

- **SmolVLA 第二相机**：checkpoint 声明 `observation.images.image` 与
  `observation.images.image2`；`vlm_stage` 现在支持后者（`extra["wrist_image"]`），
  但闸门只用合成观测，未覆盖真实 LIBERO 分布。
- **SmolVLA 后处理**：`vlm_stage` 创建了 `postprocessor`，`post_decode` 仍未调用，
  输出停在归一化空间。闸门比较的是 raw action chunk，与 vllm-omni 的 π0.5
  测试一致；用户可见的物理动作需要单独补反归一化。
- **SmolVLA noise 对齐方式**：闸门用全局 seed 让两侧各自 `sample_noise`，
  不是显式注入同一个张量；任何新增的 RNG 消费者都会让两条路径抽到不同 noise。
- **SmolVLM `pixel_attention_mask`**：非正方形 / 带 padding 的图像与 HF 不同。
  当前闸门用正方形图像，覆盖不到这条路径。
- **SmolVLM 默认采样**：`SamplingParams.temperature` 默认 1.0，HF `generate`
  默认 greedy；闸门显式传 `temperature=0`。
- **SmolVLM greedy 近似**：stage 把 `temperature=0` 映射成 `1e-5` 的 Gumbel 采样，
  不是严格 argmax；logits 接近 tie 时可能和 HF 不同。
- **MiniMind RNG 交错**：vendor 在同一循环里用 `torch.multinomial` 采样文本与
  8 个 codebook；fork 用 Gumbel-max，取数顺序不同。即使 logits 一致，采样出的
  code 也不会相同。因此音频码只做信息输出，不做闸门；P1 必须在 greedy 下比。
- **MiniMind 参考 cache 模式**：闸门两侧都用 `use_cache=False` 全序列前向
  （vendor `eval_omni.py` 默认 `use_cache=True`，带 KV cache 的 logits 有
  数值差）。
- **WSL 的 editable install 指向另一个 checkout**：`nanovllm-omni` 与
  `nano-vllm` 两个 editable 都指向 `/mnt/d/edge-inference/...`。不设
  `PYTHONPATH=$REPO:$REPO/third_party/nano-vllm` 就会跑那份旧代码（还带着
  已删的 `talker.py` 和未修的 bridge），且不会报错。闸门脚本会打印
  `repo under test` 行，跑完先看这行对不对。
- **MiniMind 音色条件**：`speaker_embedding` 从未被赋值，`ref_codes` 也不支持；
  当前固定走官方 default voice。vendor talker 的 in-place speaker mask 已在
  `_talker_logits` 实现，但接线音色前还要把 speaker token 放进 audio buffer。
- **MiniMind talker 实现**：生产路径用 vendor `MiniMindOmni.talker`（fp16），
  仓库里没有 fork talker 实现；`content_len + 64` 的 bridge tail 覆盖
  teacher-forced 实测的 56 帧。
- **MiniMind 旧 profile 数字**：在 chat template / fp16 / 早停 / vendor talker
  之前测得，需重测。
- **SD-Turbo seed**：`SamplingParams.seed` 没有接到 generator，同 seed
  不保证复现。

## 6. 非目标

- 生产 bf16 推理与 HF / 预训练 kernel 逐 bit 相同；
- 跨 GPU、跨 batch 形状、跨库版本的 bitwise；
- 用某一 dtype / device 的过闸结果推断另一配置。

## 7. 新增模型族时

先在 `.agents/skills/add-new-model/SKILL.md` 的接线清单里声明本族的
P0 / P1 / P2 闸门，再声称“与参考实现对齐”。没有闸门的对齐结论不进文档。
