# ADR 0006: 四族显存占用对比（2026-10-30）

- Status: Accepted
- Date: 2026-10-30

## 问题

四个模型族在 4 GB 的 RTX 3050 上各占多少显存？此前的记录分散在 ADR-0003
（sd-turbo / smolvlm / smolvla，2026-09-13）与 README（minimind-o，full E2E），
硬件口径相同但**代码版本、dtype、输入与 instrumentation 各不相同**，不能直接相减。
本文用同一个探针、同一台机器、同一份代码把四族拉平到一张表上。

## 环境与口径

- 机器：WSL2（`DESKTOP-Q29JAU0`），NVIDIA GeForce RTX 3050 4GB Laptop GPU，4096 MiB，driver 591.86
- 软件：python 3.12 venv，torch 2.11.0+cu130，transformers 5.16.1，diffusers 0.40.0，lerobot 0.6.1
- 代码：`perf/colab-2x` @ `78121b6`。工作树 rsync 到 WSL `~/Projects/nanovllm-omni`，
  通过 `PYTHONPATH`（含 `third_party/nano-vllm`）导入；WSL 上 `/mnt/d` 那份旧 worktree
  （`main` @ `93af9f6`，有本地改动）未被使用，也未被修改。
- 权重：本地目录 `~/models/<family>`；Mimi 用 HF 缓存里的 `kyutai/mimi`。
  全程 `HF_HUB_OFFLINE=1`，不访问网络。
- 入口：公开 seam `Omni(...).generate([prompt], SamplingParams(...))`，即 `examples/` 与
  `bench/` 走同一条路径，不直接调 stage 工厂。
- 显存口径：`torch.cuda.max_memory_allocated()`，单位 MiB。
  每次测量 = 1 次 warmup（在计时窗口外，用于触发懒加载 stage 与 JIT）＋ 3 次计时，
  取 3 次峰值的最大值；`reserved` 取 `memory_reserved()`。三次峰值完全一致，
  说明该口径下的噪声小于 0.1 MiB。

## 结果

| 族 | dtype | steady MiB | **peak MiB** | reserved MiB | wall p50 ms | 3 次是否全成功 |
|---|---|---:|---:|---:|---:|---|
| `minimind_omni` | fp16 | 990.6 | **1086.7** | 1120.0 | 7637.9 | 3/3 |
| `smolvla` | fp32（加载器固定） | 1195.1 | **1217.6** | 1242.0 | 937.8 | 3/3 |
| `smolvlm` | bf16 | 2188.9 | **2189.3** | 2270.0 | 241.4 | 3/3 |
| `sd_turbo` | fp16 | 2487.3 | **3097.6** | 3646.0 | 518.5 | 3/3 |

`steady` 是 warmup 之后的常驻占用（权重 + fork 的 paged KV 池）；`peak` 是单次请求
跑完整个 pipeline 过程中的最高分配。

> 注意：`smolvlm` 这一行的 2189 MiB 是 `d1406e5` **之前**的数（当时 stage 持有一份
> 重复模型副本，KV 池只剩 1 个 block）。修复后同机实测：加载后 1587.6 MiB，
> 文本请求峰值 1587.9 MiB，图片请求（512², 17 tiles）峰值 1922.1 MiB。
> 详见 ADR 0008。

结论：

1. **`minimind_omni` 显存占用最低**，峰值 1086.7 MiB，占 4 GiB 卡的 26.5%。
2. 次低是 `smolvla`（1217.6 MiB）。两者峰值差约 131 MiB，常驻差约 205 MiB，
   主要来自权重体量（SmolVLA 是单个 1218 MB 的 fp32 policy），而不是调度风格：
   MiniMind-O 虽然三段式，但每段都很小，且 KV 池被 `total * 0.12` 夹住。
3. `smolvlm` 与 `sd_turbo` 分别是 minimind-o 的 2.0 倍和 2.85 倍。
4. `sd_turbo` 的 `reserved` 达到 3646 MiB，距 4 GiB 上限只剩约 450 MiB；
   它是四族里唯一"再大一点就跑不动"的。

## 输入与采样配置

| 族 | 输入 | 采样 | deploy 默认值来源 |
|---|---|---|---|
| `minimind_omni` | 文本 `你好，请用一句话介绍你自己。` | `max_tokens=120`（上限，≈9.6 s 音频 @ 12.5 Hz），`seed=7` | `deploy/minimind_omni.yaml`（thinker 0.6 / talker 0.2 / code2wav 0.1） |
| `smolvla` | 合成观测：256×256 RGB + 8 维零 state | 10 个 flow step，`seed=7` | `deploy/smolvla.yaml` |
| `smolvlm` | 纯文本，无图（原因见限制 1） | `max_new_tokens=64`，`seed=7` | `deploy/smolvlm.yaml`（`gpu_memory_utilization=0.85`） |
| `sd_turbo` | 文本 `a cute cat, studio photo` | 1 step，512×512，`seed=7` | `deploy/sd_turbo.yaml` |

`gpu_memory_utilization` 对显存的影响有上限：fork 的 `allocate_paged_kv_cache`
会把 KV 池再夹到 `total * 0.12`（≈492 MiB），所以 minimind-o / smolvlm 的
KV 池都不会把卡吃满，两族的 deploy 预算差异不会扭曲这张表。

## 与旧记录的交叉验证

| 族 | 本文 | 旧记录 | 差 |
|---|---:|---:|---:|
| `smolvlm` | 2189.3 MiB | 2181 MiB（ADR-0003，3050） | +8 MiB（0.4%） |
| `sd_turbo` | 3097.6 MiB | 3098 MiB（ADR-0003，3050） | −0.4 MiB |
| `smolvla` | 1217.6 MiB | 2021 MiB（ADR-0003，3050） | −803 MiB，未定位 |

前两项对得上，说明探针与 ADR-0003 的口径一致。`smolvla` 的 803 MiB 差异**尚未解释**：
两次的代码路径、dtype（均为 fp32 policy）与显存口径相同，差异可能来自
instrumentation（ADR-0003 的 baseline 带 `torch.profiler` / Kineto 记录阶段区间，
本文不开 profiler）或权重解析路径（HF snapshot 对本地目录），未做定位。
在复跑 `bench_smolvla` 之前，本文的 1217.6 只代表"无 profiler 时的峰值"。

## 权重体积（供对照）

| 族 | 实际下载体积 | 说明 |
|---|---:|---|
| `minimind_omni` | 227 MB + 385 MB | `pytorch_model.bin`（fp32 ckpt）＋ `kyutai/mimi` |
| `smolvlm` | 1015 MB | 跳过 5.9 GB 的 `onnx/` 导出 |
| `smolvla` | 1218 MB | 单个 `model.safetensors`，489 个 tensor |
| `sd_turbo` | 2580 MB | 仅 fp16 variant（unet 1732 + text_encoder 681 + vae 167） |

## 已知限制

1. `smolvlm` 按**纯文本**测量，与 `bench_smolvlm` 及 ADR-0003 的协议一致。当前
   transformers 版本下图片路径在本仓库内不可用：processor 返回 5-D
   `pixel_values`（`1×17×3×512×512`），而 `SmolVLMStage` 直接交给 SigLIP，
   后者按 4-D 解包。视觉 token 的激活因此未计入本表；修复后需重测。
2. `minimind_omni` 的 talker 在当前环境会**间歇性**返回空 audio codes
   （`MiniMind talker audio codes must be non-empty`）。3 次连续请求即可复现：
   `seed=None` 时第 2 次失败、第 3 次成功；`seed=42` 时首轮成功、后续失败。
   本文用 `seed=7`，3 次计时全部成功且峰值三次相同。该现象与显存无关，
   但会影响任何"多次请求"的吞吐测量。
3. 主表只用 torch allocator 计数。`torch.cuda.mem_get_info()` 在本机给出
   `total - free` 与 allocator 计数相差约 840 MiB（CUDA context + NCCL 缓冲），
   而 fork 源码注释已指出 WSL 上该接口可能返回系统内存值，因此不把设备侧读数
   作为结论依据。
4. 各族 dtype 不同（fp16 / bf16 / fp32），因为那是各族**当前能跑**的默认配置；
   本文没有把它们归一到同一 dtype。`smolvla` 的 `dtype` 参数目前不被
   `models/smolvla/policy.py` 消费（lerobot 默认精度加载）。
5. 四族做的工作量不同（至多 120 帧音频 / 64 token / 1 step 512² / 10 步 flow），
   峰值显存随工作量变化。本表是"默认配置下的一次端到端请求"，不是纯权重占用，
   也不能用来比较单位工作量成本。
6. 本文的 wall 只用于说明探针确实跑通了整条链路，**不是吞吐结论**。
   `minimind_omni` 的 7.6 s 明显高于 README 的 0.65–1.04 s；探针日志里可见
   `torch.dynamo` 反复重编译（`rms_forward` 的 Half/BFloat16 dtype mismatch
   触发 `recompile_limit`），但本测未对该差异做归因。

## 复现

```bash
# 1) 同步当前工作树（不要用 /mnt/d 上的旧 worktree）
rsync -az --exclude .venv --exclude .git --exclude docs --exclude __pycache__ \
    . mcigs-wsl:~/Projects/nanovllm-omni/

# 2) 权重放到 ~/models/{minimind-3o,SmolVLM-500M-Instruct,smolvla_libero,sd-turbo}
#    Mimi 走 HF 缓存（kyutai/mimi）。两个坑：
#    - huggingface_hub 直连会停在 0 字节；用 curl 逐文件拉。
#    - 本机经 Windows 宿主代理（172.26.80.1:7897）可达 huggingface.co，
#      直连与 hf-mirror 都慢一个数量级。
```

探针本体（local-only，未入版本控制）放在本机 `.scratch/vram_probe.py`。核心循环如下，
四个族各起一个进程跑一遍即可复现上表；采样载荷与 `examples/offline_inference/`
保持一致（见上文"输入与采样配置"）：

```python
import torch
from nanovllm_omni import Omni, SamplingParams

omni = Omni(model, device="cuda", dtype=dtype, extra=extra)
with torch.no_grad():
    omni.generate([prompt], sampling)               # warmup：触发 stage 懒加载
    torch.cuda.synchronize()
    steady = torch.cuda.memory_allocated() / 2**20
    peaks = []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats()
        out = omni.generate([prompt], sampling)[0]
        torch.cuda.synchronize()
        assert not out.error
        peaks.append(torch.cuda.max_memory_allocated() / 2**20)
    print(steady, max(peaks), torch.cuda.memory_reserved() / 2**20)
```

`dtype` / `extra` / `sampling` 按族取值：`minimind_omni` 用 `float16` +
`extra={"pipeline": "minimind_o"}` + `SamplingParams(max_tokens=120, seed=7)`；
`smolvlm` 用 `bfloat16` + `extra={"images": [], "max_new_tokens": 64}`；
`sd_turbo` 用 `float16` + `extra={"num_inference_steps": 1}`；`smolvla` 不传 dtype
（加载器固定 fp32）+ `extra={"pipeline": "smolvla", "image": ..., "state": ...}`。
四族都要带 `extra={"allow_hf_download": False}`。

原始日志是本地产物，不进 git；本文只保留可复现的结论与配置。
