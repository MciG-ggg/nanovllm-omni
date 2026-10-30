"""The experiment harness turns a YAML cell into the family's bench command.

The command is the thing that actually decides what gets measured, so it is
what these tests assert. Validation is tested too: a misspelled knob must
fail loudly instead of quietly measuring the default twice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.experiment import Cell, Experiment, build_command, load_experiment

OUT = Path("/tmp/out.csv")


def test_smolvla_cells_come_from_the_yaml() -> None:
    experiment = load_experiment("smolvla")

    assert experiment.family == "smolvla"
    assert [cell.name for cell in experiment.cells] == ["eager", "compiled"]


def test_hydra_override_reaches_the_experiment() -> None:
    experiment = load_experiment("minimind", ["runs=2"])

    assert experiment.runs == 2


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        Experiment(family="sd_turbo", model="/tmp/m", cells=(Cell(name="only"),))


def test_unknown_cell_knob_is_rejected() -> None:
    with pytest.raises(ValueError, match="not understood by the smolvla bench"):
        Experiment(
            family="smolvla",
            model="/tmp/m",
            cells=(Cell(name="bad", flags={"use_cuda_graph": True}),),
        )


def test_minimind_graph_flag_follows_the_cell() -> None:
    experiment = load_experiment("minimind")

    graph = build_command(experiment, experiment.cells[0], OUT)
    eager = build_command(experiment, experiment.cells[1], OUT)

    assert "--use-thinker-cuda-graph" in graph
    assert "--no-use-thinker-cuda-graph" in eager


def test_smolvlm_eager_cell_opts_out_of_the_stage_default() -> None:
    experiment = load_experiment("smolvlm")

    graph = build_command(experiment, experiment.cells[0], OUT)
    eager = build_command(experiment, experiment.cells[1], OUT)

    assert "--no-cuda-graph" not in graph
    assert "--no-cuda-graph" in eager


def test_smolvla_compiled_cell_passes_the_flag() -> None:
    experiment = load_experiment("smolvla")

    eager = build_command(experiment, experiment.cells[0], OUT)
    compiled = build_command(experiment, experiment.cells[1], OUT)

    assert "--compile-flow" not in eager
    assert "--compile-flow" in compiled
