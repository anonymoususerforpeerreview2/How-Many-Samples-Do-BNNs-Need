"""Shared single-request timing for the three vision stopping tasks."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Callable

import numpy as np

from experiments.cifar100.evaluation import TASKS, run_tracker


def median_runtime(action: Callable[[], None], repeats: int, sync: Callable[[], None]) -> float:
    timings = []
    for _ in range(repeats):
        sync()
        start = perf_counter()
        action()
        sync()
        timings.append(perf_counter() - start)
    return float(np.median(timings))


def benchmark_fixed_budgets(
    cfg,
    count: int,
    inference_action: Callable[[np.ndarray], None],
    gpu_sync: Callable[[], None],
) -> dict[str, dict[str, float]] | None:
    """Replay quality-matched constant budgets when FIXED_BUDGETS is set.

    The comma-separated order is class selection, threshold, distribution.
    This uses the same forward-pass timer as the adaptive latency benchmark.
    """
    raw = os.environ.get("FIXED_BUDGETS")
    if raw is None:
        return None
    budgets = [int(value) for value in raw.split(",")]
    tasks = ("class_selection", "threshold", "distribution_accuracy")
    if len(budgets) != len(tasks) or any(
        budget < 1 or budget > cfg.task_max_samples(task)
        for task, budget in zip(tasks, budgets)
    ):
        raise ValueError("FIXED_BUDGETS must contain three valid budgets")
    rows = {}
    for task, budget in zip(tasks, budgets):
        schedule = np.full(count, budget, dtype=np.int64)
        seconds = median_runtime(
            lambda: inference_action(schedule), cfg.runtime_repeats, gpu_sync
        ) / count
        rows[task] = {"draws": budget, "inference_seconds": seconds}
        print(f"fixed {task}: {rows[task]}", flush=True)
    return rows


def benchmark_tasks(
    cfg,
    samples: np.ndarray,
    result_path,
    inference_action: Callable[[np.ndarray], None],
    gpu_sync: Callable[[], None],
) -> dict[str, dict[str, float]]:
    """Time forward passes and the CS separately on every clean input.

    Stop times come from the saved evaluation stream. The separate inference
    timing replays those budgets; it is not an online, interleaved timing.
    """
    with np.load(result_path) as results:
        schedules = {
            task: np.asarray(results[f"{task}_stop_times"], dtype=np.int64)
            for task in TASKS
        }
    count = samples.shape[1]
    if count != cfg.num_inputs or any(len(values) != count for values in schedules.values()):
        raise ValueError(f"expected {cfg.num_inputs} clean inputs in both saved artifacts")

    rows = {}
    for task in TASKS:
        schedule = schedules[task]
        if np.any(schedule < 1) or np.any(schedule > cfg.task_max_samples(task)):
            raise ValueError(f"invalid saved stopping times for {task}")

        inference_seconds = median_runtime(
            lambda: inference_action(schedule), cfg.runtime_repeats, gpu_sync
        ) / count

        def cs_action() -> None:
            for index in range(count):
                stream = np.asarray(samples[: cfg.task_max_samples(task), index : index + 1])
                result = run_tracker(task, stream, cfg)
                if int(result.n_samples[0]) != int(schedule[index]):
                    raise RuntimeError(f"saved and replayed {task} stopping times disagree")

        cs_seconds = median_runtime(cs_action, cfg.cs_repeats, lambda: None) / count
        rows[task] = {
            "mean_draws": float(np.mean(schedule)),
            "inference_seconds": inference_seconds,
            "cs_seconds": cs_seconds,
            "total_seconds": inference_seconds + cs_seconds,
        }
        print(f"{task}: {rows[task]}", flush=True)
    return rows
