# ADR 0023: MLX 轨的执行顺序——thinker-only 起步，kernel 先 decode

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0013（backend 接缝）、0014（双轨目标）、0021（kernel 自研最小集、
  token 级一致）、0022（CUDA 轨这一轮）

## 决定

### 第一站：MiniMind-O 的 thinker（不含 talker）

用 `mlx_lm` 转出来的 MiniMind-O thinker 跑通 MLX backend 的**完整 AR 侧**：
varlen attention、批 decode、slot 复用。

**为什么不是 SmolVLM**：

- MiniMind-O 的 thinker 是 `Qwen3ForCausalLM` 契约（`thinker.py:247` 自述），
  `mlx_lm` 直接能转权重。
- SmolVLM 的视觉塔是 SigLIP（第三方库，ADR 0020 定的边界），要走 MLX 版
  得先移植它——为一个"验证 AR 侧"的目标多付一份移植。

**talker 断链不影响这个第一站**：验证 varlen kernel + backend + 批处理只需要 AR 侧完整。
talker 的 MLX 版是后面的事，它的前置是 ADR 0017 的重写（重写之后才谈得上写 MLX 版）。

### kernel 顺序：先 decode，再 prefill

- **decode** 最简单：每个序列 1 个 query，k/v 按 block table 从 KV 里读，
  按各序列的 context_len 屏蔽。没有扁平拼接、没有 `cu_seqlens`。
- **prefill** 才是 varlen 的主要收益所在（齐长批 vs ragged 批差 143%，
  ADR 0015 引的上游实测），但要处理扁平拼接 + 边界划界。

先做 decode 是为了让 kernel 有一条能验证的最短路径：decode 的正确性可以直接对着
CUDA 轨的 `audio_codes` / token 序列比（ADR 0021 的 token 级一致）。

## 后果

- **第一站的验收**：MLX 上 MiniMind-O thinker 能出与 CUDA 轨**同 token 序列**的结果
  （相同权重、seed、采样配置），并能跑一个多序列的批。
- **与 CUDA 轨的关系是趋势可比，绝对值不可比**（ADR 0014：不跨机相减）。
  比较的是"齐长批 vs ragged 批的相对变化"，不是"谁更快"。
- **第一站不覆盖** talker、code2wav、视觉、diffusers、lerobot。MLX 轨的覆盖面
  天然小于 CUDA 轨，这是 ADR 0014 已经承认的边界。
- **采样配置必须严格对齐**才能拿到 token 级一致（ADR 0021）：
  temperature / top-k / top-p / seed / max_tokens 一个都不能差。

## 边界

- 本 ADR **不**承诺 MLX 轨能跑完整 omni 管线。MiniMind-O 的完整管线需要 talker，
  而 talker 的 MLX 版尚未存在。
- 本 ADR **不**决定 kernel 源码放哪个目录、怎么编译加载——实现细节。
- 本 ADR **不**改 ADR 0019 的分工：大块归 MLX 轨，本 ADR 定的正是 MLX 轨的内部顺序。
