# ADR 0020: 「自己写」的边界——只重写 vendor 自定义代码

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0017（talker 重写 + 窗口化 KV）、0016（serving 是教学对象）

## 问题

ADR 0017 定了 talker 的模型代码自己写。往下一层的问题是：**"自己写"的边界到哪？**

这个仓库现有的外部模型代码依赖，实测盘点如下：

| 部件 | 外部代码来源 | 性质 |
|---|---|---|
| talker | checkpoint 目录的 `model_omni.py`（`get_class_from_dynamic_module` 动态加载） | **vendor 自定义代码**——这个模型专属 |
| code2wav | `transformers` 的 `MimiModel` + `kyutai/mimi` 权重 | 第三方库 |
| smolvlm 视觉侧 | `transformers` 的 SigLIP | 第三方库 |
| smolvlm 语言侧 | 本仓库自研 `SmolLM2*`（`models/smolvlm/smolvlm.py`） | 已经是自己的 |
| sd_turbo | `diffusers`：CLIPTextModel / UNet2DConditionModel / AutoencoderKL / EulerDiscreteScheduler | 第三方库 |
| smolvla | `lerobot` 的 `SmolVLAPolicy` | 第三方库 |

按字面理解"除了 fork 的 nanovllm 都要自己写"，会包含重写 SigLIP、Mimi、
diffusers 的 UNet/VAE、lerobot 的 policy——这是以月计的工程量，而且会毁掉
"教学一条真实 pipeline"这件事本身：读者来看这个仓库是为了看**真实的工程取舍**，
不是为了看一份从零手写的 UNet。

## 决定

**只重写 vendor 自定义代码（当前只有 talker）。第三方库照用。**

分界线是**代码归属**，不是复杂度：

- **vendor 代码**是这个模型的专属实现，随 checkpoint 一起变。它不进 git 就没有版本、
  不可读、不能改——talker 的窗口化 KV 就无从谈起。**必须重写。**
- **第三方库**是别人维护的通用实现，有自己的发布节奏与语义。照用是正确的工程选择，
  重写它既没有教学价值也没有工程价值。

## 后果

- **checkpoint 目录的角色变了**：从"代码 + 权重"变成"**权重 + parity 参照**"。
  vendor 代码**不删**——ADR 0017 要求重写版与 vendor 输出逐位一致，
  需要一份可运行的参照实现来生成基线。差别是：它是参照物，不是运行依赖。
- **其余三族不受影响**，它们的外部依赖照旧。MLX 轨上它们各自另写 MLX 版
  （那是 MLX 轨的事，与本 ADR 的"重写 vendor"不是一回事——见 ADR 0021 的边界）。
- **本 ADR 不禁止将来重写第三方库**，只是现在不做。真要重写某一族时，
  要单开一个 ADR 说明为什么"照用"这次不够。

## 边界

- talker 的重写范围是 `TalkerModule`（embed / layers / lm_head / freqs 表），
  不含 `code2wav` 用的 `MimiModel`——那是 transformers 的模型类，属第三方库。
- 本 ADR 只管"哪些代码归我们写"，不管"代码放在哪个文件"。
