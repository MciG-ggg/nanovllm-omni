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
| MiniMind-O | `tests/test_parity_minimind.py` → `tests/test_minimind_hf_parity.py` | P1（文本）+ P2（bridge） | `PYTHONPATH=<repo>:<repo>/third_party/nano-vllm pytest -m smoke tests/test_parity_minimind.py` | **已跑通（2026-10-31）**：P1 greedy 文本 20/20 逐 token 相等；P2 bridge 47 行 cosine 1.00000000、mean relative 0.0517%。参照栈见 §4.1 |

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
| P2 | bridge 行（layer 3 输出） | cosine 1.00000000，mean relative 0.05%，最大元素差 1.0（在 ~1004 量级上约 1 个 fp16 ULP） |
| P2 | talker 码本 argmax（teacher-forced） | 124/124 = 100% |
| P2 | talker 采样码（同 seed、teacher-forced） | 123/124 = 99.2%（唯一分歧是 124 个位置里的一次采样翻转，argmax 相同） |

结论：fork thinker（paged attention + 融合 kernel + fp16）与 vendor 的 eager
全序列实现在 fp16 舍入内等价。分歧只剩 RNG：两侧都调 `torch.multinomial`，
但 vendor 在同一个循环里交替抽文本 token 和 8 个 codebook，而本仓库把 talker
当成独立阶段跑（中间还会重新播种），两条路径消耗 RNG 流的位置不同，
**采样路径下的 code 相同不是预期**。
旧版闸门把两侧**采样后**的音频码逐位置比较（阈值 87.5%），实测 7.81%——
那不是实现错误，是 RNG 流的错位。闸门已改成上面的 P1/P2，采样码不做判定。

本小节里：P1 与 P2 bridge 两行是**闸门脚本自己产出**的；P2 文本 logits
和 §4.2 的 talker 两行来自一次性 `/tmp` harness，闸门不计算它们。
所有数字来自同一个 prompt、seed 42、20 个生成 token：它是一条记录，
不是跨 prompt 的统计结论。

### 4.2 talker 的 teacher-forced 对照（2026-10-31）

生产链路的 talker 就是 vendor 的 `TalkerModule`（fp16，`strict=True` 加载
`talker.*`），只有驱动循环是本仓库的（`TalkerStage._talker_logits` + 交错
调度 + 采样）。所以 talker 侧唯一的不确定是：**我们的 bridge 比 vendor
差 0.05%，会不会把音频码采错。**

两臂用**同一个** `TalkerModule`、走**生产驱动** `_talker_logits`，码历史
teacher-forced 自 vendor 臂（误差不累积），只有 bridge 不同。20 步 × 8 码本，
其中真实采样位置 124 个（码本 `i` 从第 `i+1` 步才开始采）：

| 度量 | 结果 |
|---|---|
| audio logits max abs diff | 0.0625 |
| audio logits mean abs diff | 0.0047 |
| 码本 argmax 一致（真实位置） | 124/124 = 100% |
| 同 seed 采样码一致（真实位置） | 123/124 = 99.2% |

即：thinker 的 0.05% bridge 误差不改变 talker 的 argmax，只在一个位置翻了一个
采样。这是当前“音频是人话”的数值解释。

这条目前是**一次性 harness**（`/tmp`，未入库），不是闸门。要升级为闸门需要把
它写成 P2 测试；在那之前，ADR 不声称 talker 有自动闸门。

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
- **MiniMind RNG 交错**：两侧都调 `torch.multinomial`，但 vendor 在同一个
  循环里交替抽文本 token 与 8 个 codebook，而本仓库把 talker 当独立阶段跑，
  消耗 RNG 流的位置不同。即使 logits 一致，采样出的 code 也不会相同。
  因此音频码只做信息输出，不做闸门；P1 必须在（近似）greedy 下比。
- **MiniMind 参考 cache 模式**：vendor 侧用 `use_cache=False` 全序列前向
  （官方 `eval_omni.py` 默认 `use_cache=True`，带 KV cache 的 logits 有
  数值差）。**本仓库侧并不都是全序列**：bridge 行是 `recapture_bridge_minimind`
  重跑一次全序列 prefill 得到的，而 P1 的文本 token 走的是 paged KV 增量
  decode（`decode_minimind`）。两侧唯一共同的输入只有 token 序列。
- **WSL 的 editable install 指向另一个 checkout**：`nanovllm-omni` 与
  `nano-vllm` 两个 editable 都指向 `/mnt/d/edge-inference/...`。不设
  `PYTHONPATH=$REPO:$REPO/third_party/nano-vllm` 就会跑那份旧代码（还带着
  已删的 `talker.py` 和未修的 bridge），且不会报错。闸门脚本会打印
  `repo under test` 行，跑完先看这行对不对。
- **MiniMind 音色条件**：`speaker_embedding` 从未被赋值，`ref_codes` 也不支持；
  当前固定走官方 default voice。vendor talker 的 in-place speaker mask 已在
  `_talker_logits` 实现，但接线音色前还要把 speaker token 放进 audio buffer。
- **MiniMind 与官方全模型的帧数差距尚未定因**：同一 prompt 上，官方
  `MiniMindOmni`（`use_cache=True`、官方采样）产出了 `minimind_hf_reference.wav`
  72 帧 / 5.76 s、18 个生成 token；本仓库产出 55 帧 / 4.40 s、16 个 token。
  这**不是**同一条语音：两边采样出的文本就不同，所以帧数不可直接相减。
  同输入的唯一对照是 teacher-forced（同一 16 token）：官方 talker 56 帧、
  我们 55 帧，差的 1 帧来自采样码不同导致的 stop token 位置差一步。
  72 → 55 那 17 帧的完整解释（文本长度比例 18:16 只能解释一部分）未查清，
  不要引用“我们和官方只差 1 帧”这种说法。
- **MiniMind talker 实现**：生产路径用 vendor `MiniMindOmni.talker`（fp16），
  仓库里没有 fork talker 实现；`content_len + 64` 的 bridge tail 覆盖
  teacher-forced 实测的 56 帧。
- **MiniMind talker 没有自动闸门**：§4.2 的 teacher-forced 对照是一次性
  harness，未入库。ADR 只在 P1（文本）和 P2（bridge）上声称有闸门。
- **MiniMind profile 数字**：已重测（2026-10-31，WSL / RTX 3050，bench prompt
  `short_01`，`bench time --pipeline full --max-tokens 512`）：wall 1539.4 ms、
  thinker 473.2、talker 1039.4、code2wav 26.8（37 帧）。详见 ADR 0007。
- **SD-Turbo seed**：`SamplingParams.seed` 没有接到 generator，同 seed
  不保证复现。

## 6. 非目标

- 生产 bf16 推理与 HF / 预训练 kernel 逐 bit 相同；
- 跨 GPU、跨 batch 形状、跨库版本的 bitwise；
- 用某一 dtype / device 的过闸结果推断另一配置。

## 7. 新增模型族时

先在 `.agents/skills/add-new-model/SKILL.md` 的接线清单里声明本族的
P0 / P1 / P2 闸门，再声称“与参考实现对齐”。没有闸门的对齐结论不进文档。
