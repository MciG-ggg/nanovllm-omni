"""CSV writer + markdown table renderer for RunResult rows."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

CSV_COLUMNS: tuple[str, ...] = (
    "prompt",
    "tokenize_ms",
    "generate_ms",
    "decode_ms",
    "wav_ms",
    "total_ms",
    "n_tokens",
    "n_samples",
    "max_mem_bytes",
    "wav_bytes",
)


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write rows to ``path`` as CSV; parent dirs are created if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    return p


def markdown_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str] | None = None,
) -> str:
    """Render rows as a GitHub-flavored markdown table."""
    cols = list(columns) if columns is not None else list(CSV_COLUMNS)
    if not rows:
        return "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    out: list[str] = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        cells = [_fmt(row.get(c, "")) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
