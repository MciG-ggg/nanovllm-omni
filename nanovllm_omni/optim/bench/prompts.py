"""Fixed prompt set for the TK-011 Session-1 baseline.

Six prompts covering three lengths and a system scene, chosen so the
four-stage breakdown (tokenize / generate / decode / wav) shows meaningful
variance without exceeding the 4 GB RTX 3050.
"""

from __future__ import annotations

BENCH_PROMPTS: tuple[str, ...] = (
    # 3 short (< 50 zh chars)
    "你好。",
    "今天天气真好。",
    "谢谢。",
    # 2 medium (100-300 zh chars)
    "请用两句话介绍MiniMind-O模型的优势和局限，并解释它和传统语言模型的主要区别。",
    "想象你在一个安静的咖啡馆里，请描述你看到的场景、听到的声音以及你内心的感受，字数在两百字左右。",
    # 1 system scene
    "[系统提示]你是一个温柔的儿童故事讲述者，正在给五岁的小朋友讲睡前故事。请用简单温暖的语言讲述一个关于月亮和兔子的故事。",
)
