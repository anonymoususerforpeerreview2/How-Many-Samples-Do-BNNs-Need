"""Synthetic three-class experiments shared by paper figure scripts."""

from .core import (
    TASK_LABELS,
    ToyData,
    ToyBaselineResult,
    ToyTaskResult,
    load_config,
    make_toy_data,
    paper_example_indices,
    paper_example_stopping_times,
    run_task,
    run_baselines,
    write_tex_macros,
)

__all__ = [
    "TASK_LABELS",
    "ToyData",
    "ToyBaselineResult",
    "ToyTaskResult",
    "load_config",
    "make_toy_data",
    "paper_example_indices",
    "paper_example_stopping_times",
    "run_task",
    "run_baselines",
    "write_tex_macros",
]
