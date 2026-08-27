"""Per-stage kernel breakdown parsed from a torch.profiler Kineto trace."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGE_NAMES = frozenset({"tokenize", "generate", "decode", "wav"})
# Sub-event names we want to ignore when counting "child kernels" of a stage.
_SUB_STAGE_NAMES = frozenset({"generate.step"})


@dataclass(frozen=True)
class KernelStat:
    """Aggregated stats for one CUDA kernel inside a stage."""

    name: str
    total_us: float
    count: int


@dataclass(frozen=True)
class StageProfile:
    """Per-stage breakdown extracted from the Kineto trace."""

    name: str
    wall_us: float
    kernel_count: int
    total_kernel_us: float
    top_kernels: tuple[KernelStat, ...] = field(default_factory=tuple)
    # n_steps shows how many AR iterations ran inside generate (informational
    # when the trace is from a single generate call).
    num_steps: int = 0


@dataclass(frozen=True)
class TraceProfile:
    stages: tuple[StageProfile, ...]

    def by_stage(self, name: str) -> StageProfile | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None


def parse_kineto_trace(json_path: str | Path) -> TraceProfile:
    """Parse a torch.profiler Chrome trace and group kernel events by stage."""
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    events = data.get("traceEvents", [])
    return _parse_events(events)


def _parse_events(events: list[dict[str, Any]]) -> TraceProfile:
    stage_events = [
        ev
        for ev in events
        if ev.get("ph") == "X"
        and ev.get("cat") == "user_annotation"
        and ev.get("name") in STAGE_NAMES
    ]

    profiles: list[StageProfile] = []
    for stage in stage_events:
        s_start = stage["ts"]
        s_end = s_start + stage["dur"]
        kernel_dur: dict[str, int] = defaultdict(int)
        kernel_count: dict[str, int] = defaultdict(int)
        total_kernel_us = 0
        kernel_n = 0
        num_steps = 0
        for ev in events:
            if ev.get("ph") != "X":
                continue
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            if ts < s_start or ts + dur > s_end:
                continue
            if ev is stage:
                continue
            cat = ev.get("cat", "")
            name = ev.get("name", "")
            # Count generate.step sub-events so the trace tells us how many
            # AR iterations the model ran inside this generate stage.
            if name == "generate.step" and stage["name"] == "generate":
                num_steps += 1
                continue
            if name in STAGE_NAMES:
                continue
            if cat not in ("kernel", "cuda_runtime"):
                continue
            kernel_dur[name] += dur
            kernel_count[name] += 1
            total_kernel_us += dur
            kernel_n += 1
        top = sorted(kernel_dur.items(), key=lambda kv: -kv[1])[:5]
        top_kernels = tuple(KernelStat(name=k, total_us=v, count=kernel_count[k]) for k, v in top)
        profiles.append(
            StageProfile(
                name=stage["name"],
                wall_us=stage["dur"],
                kernel_count=kernel_n,
                total_kernel_us=total_kernel_us,
                top_kernels=top_kernels,
                num_steps=num_steps,
            )
        )
    return TraceProfile(stages=tuple(profiles))


def trace_profile_markdown(profile: TraceProfile) -> str:
    """Render the per-stage kernel summary as a markdown table."""
    cols = (
        "stage",
        "wall_us",
        "n_steps",
        "kernel_count",
        "total_kernel_us",
        "top_kernel",
        "top_kernel_us",
    )
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for s in profile.stages:
        top = s.top_kernels[0] if s.top_kernels else None
        lines.append(
            "| "
            + " | ".join(
                (
                    s.name,
                    f"{s.wall_us:.1f}",
                    str(s.num_steps),
                    str(s.kernel_count),
                    f"{s.total_kernel_us:.1f}",
                    top.name if top else "(none)",
                    f"{top.total_us:.1f}" if top else "-",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def trace_profile_top_kernels(profile: TraceProfile, *, per_stage: int = 5) -> str:
    """Render the per-stage top-N kernels as a markdown table per stage."""
    parts: list[str] = []
    for s in profile.stages:
        parts.append(f"### {s.name}  (wall={s.wall_us:.1f}us, kernels={s.kernel_count})")
        if not s.top_kernels:
            parts.append("\n_no kernels captured_\n")
            continue
        parts.append("\n| kernel | total_us | count |")
        parts.append("| --- | ---: | ---: |")
        for k in s.top_kernels[:per_stage]:
            parts.append(f"| {k.name} | {k.total_us:.1f} | {k.count} |")
        parts.append("")
    return "\n".join(parts)
