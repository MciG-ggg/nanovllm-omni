"""CSV writer + markdown table renderer for RunResult rows (TK-011)."""

from __future__ import annotations

import csv
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path

from .runner import RunResult

CSV_COLUMNS: tuple[str, ...] = (
    # Spec columns (issue #23).
    "prompt_id",
    "seed",
    "run_idx",
    "tokenize_ms",
    "generate_ms",
    "decode_ms",
    "wav_ms",
    "total_ms",
    "frames",
    "vram_mb",
    # Per-stage detail (TK-011 follow-up).
    "generate_cuda_ms",
    "decode_cuda_ms",
    "cpu_dispatch_ms",
    "generate_per_step_ms",
    # E2E per-stage wall-clock JSON (full-pipeline bench only; empty for thinker).
    "stage_ms",
)


def write_csv(results: Iterable[RunResult], path: str | Path) -> Path:
    """Write RunResult rows to ``path`` as CSV; parent dirs are created if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_csv_row())
    return p


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def markdown_table(results: Sequence[RunResult]) -> str:
    """Per-prompt aggregate table per spec.

    Columns: ``prompt_id | tokenize_median | generate_median | decode_median
    | wav_median | total_median | total_p95 | total_min | frames | vram_mb``.
    """
    by_prompt: dict[str, list[RunResult]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_id, []).append(r)

    cols = (
        "prompt_id",
        "tokenize_median",
        "generate_median",
        "decode_median",
        "wav_median",
        "total_median",
        "total_p95",
        "total_min",
        "frames",
        "vram_mb",
    )

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for prompt_id in sorted(by_prompt):
        rs = by_prompt[prompt_id]
        token = [r.times.tokenize_ms for r in rs]
        gen = [r.times.generate_ms for r in rs]
        dec = [r.times.decode_ms for r in rs]
        wav = [r.times.wav_ms for r in rs]
        tot = [r.times.total_ms for r in rs]
        frames = [float(r.frames) for r in rs]
        vram = [r.vram_peak_mb for r in rs]
        cells = (
            prompt_id,
            f"{(statistics.median(token) if token else 0.0):.2f}",
            f"{(statistics.median(gen) if gen else 0.0):.2f}",
            f"{(statistics.median(dec) if dec else 0.0):.2f}",
            f"{(statistics.median(wav) if wav else 0.0):.2f}",
            f"{(statistics.median(tot) if tot else 0.0):.2f}",
            f"{_percentile(tot, 95):.2f}",
            f"{min(tot):.2f}",
            f"{int(statistics.median(frames) if frames else 0.0)}",
            f"{(statistics.median(vram) if vram else 0.0):.2f}",
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def markdown_table_detail(results: Sequence[RunResult]) -> str:
    """Extended per-prompt aggregate that includes per-stage GPU/CPU breakdown.

    Adds columns for ``generate_cuda_median``, ``decode_cuda_median``,
    ``cpu_dispatch_median`` and ``generate_per_step_median`` on top of the
    spec-compliant ``markdown_table``.
    """
    by_prompt: dict[str, list[RunResult]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_id, []).append(r)

    cols = (
        "prompt_id",
        "generate_median",
        "decode_median",
        "total_median",
        "generate_cuda_median",
        "decode_cuda_median",
        "cpu_dispatch_median",
        "generate_per_step_median",
        "frames",
        "vram_mb",
    )

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for prompt_id in sorted(by_prompt):
        rs = by_prompt[prompt_id]
        gen = [r.times.generate_ms for r in rs]
        dec = [r.times.decode_ms for r in rs]
        tot = [r.times.total_ms for r in rs]
        gen_cuda = [r.times.generate_cuda_ms for r in rs]
        dec_cuda = [r.times.decode_cuda_ms for r in rs]
        cpu_disp = [r.times.cpu_dispatch_ms for r in rs]
        per_step = [r.times.generate_ms / r.frames if r.frames else 0.0 for r in rs]
        frames = [float(r.frames) for r in rs]
        vram = [r.vram_peak_mb for r in rs]
        cells = (
            prompt_id,
            f"{(statistics.median(gen) if gen else 0.0):.2f}",
            f"{(statistics.median(dec) if dec else 0.0):.2f}",
            f"{(statistics.median(tot) if tot else 0.0):.2f}",
            f"{(statistics.median(gen_cuda) if gen_cuda else 0.0):.2f}",
            f"{(statistics.median(dec_cuda) if dec_cuda else 0.0):.2f}",
            f"{(statistics.median(cpu_disp) if cpu_disp else 0.0):.2f}",
            f"{(statistics.median(per_step) if per_step else 0.0):.2f}",
            f"{int(statistics.median(frames) if frames else 0.0)}",
            f"{(statistics.median(vram) if vram else 0.0):.2f}",
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
