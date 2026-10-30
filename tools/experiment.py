"""Config-driven experiment harness: Hydra composes the cells, tyro is the CLI.

An experiment is a YAML under ``tools/experiments/``: one family, one model,
and a list of cells. Each cell runs in its own process, because the switches
it toggles (``enforce_eager``, ``compile_flow``) are fixed when a stage is
built — two cells in one process would measure the first cell's choice.

    python tools/experiment.py --experiment smolvla
    python tools/experiment.py --experiment minimind --overrides runs=2 --only eager

The harness owns no measurement logic: each cell is the family's existing
bench entrypoint, invoked with flags derived from the cell. Adding an
experiment is a YAML edit; adding a knob is one entry in ``_ALLOWED_FLAGS``
plus one line in that family's command builder.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"

# Which knobs each family's bench understands. A YAML key outside this set is
# an error rather than a silent no-op, so a typo cannot produce a "cell" that
# measures the default twice.
_ALLOWED_FLAGS: dict[str, frozenset[str]] = {
    "minimind": frozenset({"use_thinker_cuda_graph"}),
    "smolvlm": frozenset({"use_cuda_graph"}),
    "smolvla": frozenset({"compile_flow"}),
}


@dataclass(frozen=True)
class Cell:
    name: str
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    family: str
    model: str
    cells: tuple[Cell, ...]
    runs: int = 5
    warmup: int = 1
    device: str = "cuda"
    max_tokens: int = 120
    seed: int = 7

    def __post_init__(self) -> None:
        if self.family not in _ALLOWED_FLAGS:
            raise ValueError(f"unknown family {self.family!r}; known: {sorted(_ALLOWED_FLAGS)}")
        if not self.cells:
            raise ValueError("an experiment needs at least one cell")
        allowed = _ALLOWED_FLAGS[self.family]
        for cell in self.cells:
            unknown = set(cell.flags) - allowed
            if unknown:
                raise ValueError(
                    f"cell {cell.name!r}: {sorted(unknown)} not understood by the "
                    f"{self.family} bench; allowed: {sorted(allowed)}"
                )


def load_experiment(name: str, overrides: list[str] | tuple[str, ...] = ()) -> Experiment:
    """Compose ``tools/experiments/<name>.yaml`` with Hydra and validate it."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(EXPERIMENTS_DIR), version_base=None):
        config = compose(config_name=name, overrides=list(overrides))
    data = OmegaConf.to_container(config, resolve=True)

    return Experiment(
        family=str(data["family"]),
        model=str(data["model"]),
        cells=tuple(
            Cell(name=str(c["name"]), flags=dict(c.get("flags") or {})) for c in data["cells"]
        ),
        runs=int(data.get("runs", 5)),
        warmup=int(data.get("warmup", 1)),
        device=str(data.get("device", "cuda")),
        max_tokens=int(data.get("max_tokens", 120)),
        seed=int(data.get("seed", 7)),
    )


def build_command(experiment: Experiment, cell: Cell, out_path: Path) -> list[str]:
    """The family's bench entrypoint, with this cell's flags applied."""
    common = [
        "--model",
        experiment.model,
        "--device",
        experiment.device,
        "--runs",
        str(experiment.runs),
        "--warmup",
        str(experiment.warmup),
        "--out",
        str(out_path),
    ]
    if experiment.family == "minimind":
        command = [
            sys.executable,
            "-m",
            "nanovllm_omni.bench",
            "time",
            "--pipeline",
            "full",
            *common,
            "--max-tokens",
            str(experiment.max_tokens),
            "--seed",
            str(experiment.seed),
        ]
        graph = cell.flags.get("use_thinker_cuda_graph")
        if graph is not None:
            command.append("--use-thinker-cuda-graph" if graph else "--no-use-thinker-cuda-graph")
        return command
    if experiment.family == "smolvlm":
        command = [sys.executable, "-m", "nanovllm_omni.bench.bench_smolvlm", *common, "--no-trace"]
        if cell.flags.get("use_cuda_graph") is False:
            command.append("--no-cuda-graph")
        return command
    command = [sys.executable, "-m", "nanovllm_omni.bench.bench_smolvla", *common, "--no-trace"]
    if cell.flags.get("compile_flow"):
        command.append("--compile-flow")
    return command


def run_experiment(experiment: Experiment, out_dir: Path, only: tuple[str, ...] = ()) -> int:
    """Run each selected cell in a fresh process; return the failure count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for cell in experiment.cells:
        if only and cell.name not in only:
            continue
        out_path = out_dir / f"{experiment.family}-{cell.name}.csv"
        command = build_command(experiment, cell, out_path)
        print(f"\n$ {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=str(REPO_ROOT))
        if result.returncode:
            failures += 1
            print(f"[{cell.name}] FAILED with {result.returncode}", file=sys.stderr, flush=True)
        else:
            print(f"[{cell.name}] -> {out_path}", flush=True)
    return failures


@dataclass(frozen=True)
class Cli:
    """CLI surface; Hydra overrides ride along as ``key=value`` strings."""

    experiment: str = "smolvla"
    only: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()
    out_dir: str = ""


def main(argv: list[str] | None = None) -> int:
    import tyro

    cli = tyro.cli(Cli, args=argv)
    experiment = load_experiment(cli.experiment, cli.overrides)
    out_dir = (
        Path(cli.out_dir)
        if cli.out_dir
        else REPO_ROOT / "docs" / "perf" / "experiments" / cli.experiment
    )

    print(f"experiment={cli.experiment} cells={[c.name for c in experiment.cells]}")
    return run_experiment(experiment, out_dir, cli.only)


if __name__ == "__main__":
    raise SystemExit(main())
