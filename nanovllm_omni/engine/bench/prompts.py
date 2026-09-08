"""Fixed prompt set for the TK-011 Session-1 baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchPrompt:
    """One prompt in the benchmark suite.

    ``id`` is the stable identifier used by the CLI (``--prompts=short_01,...``).
    ``text`` is the user message; ``system`` carries an optional system prompt
    for prefix-cache-relevant scenarios (TK-014).
    """

    id: str
    text: str
    system: str | None = None


BENCH_PROMPTS: tuple[BenchPrompt, ...] = (
    # 3 short (< 50 zh chars)
    BenchPrompt(id="short_01", text="你好。"),
    BenchPrompt(id="short_02", text="今天天气真好。"),
    BenchPrompt(id="short_03", text="谢谢。"),
    # 2 medium (100-300 zh chars)
    BenchPrompt(
        id="medium_01",
        text=("请用两句话介绍MiniMind-O模型的优势和局限，" "并解释它和传统语言模型的主要区别。"),
    ),
    BenchPrompt(
        id="medium_02",
        text=(
            "想象你在一个安静的咖啡馆里，请描述你看到的场景、"
            "听到的声音以及你内心的感受，字数在两百字左右。"
        ),
    ),
    # 1 system scene (system prompt + short user message)
    BenchPrompt(
        id="system_01",
        text="讲一个关于月亮和兔子的睡前故事。",
        system=(
            "你是一个温柔的儿童故事讲述者，正在给五岁的小朋友讲睡前故事。"
            "请用简单温暖的语言回答。"
        ),
    ),
)
