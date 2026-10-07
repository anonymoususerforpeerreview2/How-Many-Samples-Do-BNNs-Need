"""Benchmark helpers for the paper's bounding-box implementation ablation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
from multiprocessing import get_context
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Tuple

import adaptive_cs as acs
import numpy as np
from non_vectorized_class_selection import (
    ConfidenceSequenceSettings,
    ClassSelection as ReferenceClassSelection,
)

CSBuilder = Callable[..., object]


def _time_bounds(builder: CSBuilder, draws: np.ndarray) -> Tuple[float, float]:
    """Time per-draw updates and validate the final bounding box."""
    initialization_start = perf_counter()
    tracker = builder()
    initialization_seconds = perf_counter() - initialization_start

    update_start = perf_counter()
    result = None
    for draw in draws:
        result = tracker.update(draw)
    update_seconds = perf_counter() - update_start
    assert result is not None
    lower = np.asarray(result.lower)
    upper = np.asarray(result.upper)
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise RuntimeError("The confidence sequence produced non-finite bounds.")
    return initialization_seconds, update_seconds


def _time_batch_bounds(builder: CSBuilder, trajectories: list[np.ndarray]) -> Tuple[float, float]:
    """Time one-draw-at-a-time updates for all benchmark inputs.

    The C++ tracker can consume a multi-draw block while computing only
    the final bounds. That mode is useful in production, but would make this
    ablation unfair because the conventional reference checks every draw.
    Feed the C++ tracker one draw at a time here so every variant performs the
    same number of bound reconstructions.
    """
    initialization_start = perf_counter()
    tracker = builder()
    initialization_seconds = perf_counter() - initialization_start
    draws = np.stack(trajectories, axis=1)
    update_start = perf_counter()
    result = None
    for draw in draws:
        result = tracker.update(draw)
    update_seconds = perf_counter() - update_start
    assert result is not None
    lower = result.lower.numpy()
    upper = result.upper.numpy()
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise RuntimeError("The confidence sequence produced non-finite bounds.")
    return initialization_seconds, update_seconds


def _affinity() -> tuple[int, ...] | None:
    return tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else None


def _restore_affinity(cpus: tuple[int, ...] | None) -> None:
    if cpus is not None:
        os.sched_setaffinity(0, cpus)


def _time_reference_task(
    task: tuple[ConfidenceSequenceSettings, np.ndarray]
) -> tuple[float, int | None]:
    settings, draws = task
    _, seconds = _time_bounds(lambda: ReferenceClassSelection(settings), draws)
    cpus = _affinity()
    return seconds, len(cpus) if cpus is not None else None


_ABLATION_CONFIGS = (
    ("reference", "Conventional Python", "reference", "full", "bisection"),
    ("cpp-bisection", "C++", "cpp", "full", "bisection"),
    ("cpp-bisection-componentwise", "C++ (+ componentwise)", "cpp", "componentwise", "bisection"),
    ("cpp-brent-componentwise", "C++\n(Brent + componentwise)", "cpp", "componentwise", "brent"),
    ("cpp-boost-componentwise", "C++ (TOMS 748 + componentwise)", "cpp", "componentwise", "boost"),
)


def _select_ablation_indices(
        is_hard: np.ndarray,
        *,
        per_difficulty: int,
        seed: int,
) -> np.ndarray:
    if per_difficulty < 1:
        raise ValueError("per_difficulty must be positive.")
    hard_indices = np.flatnonzero(is_hard)
    easy_indices = np.flatnonzero(~is_hard)
    if not len(hard_indices) or not len(easy_indices):
        raise RuntimeError(
            "Ablation requires both easy and hard examples in the toy dataset."
        )
    rng = np.random.default_rng(seed + 40_000)
    selected_hard = rng.choice(
        hard_indices,
        size=min(per_difficulty, len(hard_indices)),
        replace=False,
    )
    selected_easy = rng.choice(
        easy_indices,
        size=min(per_difficulty, len(easy_indices)),
        replace=False,
    )
    return np.concatenate((selected_easy, selected_hard)).astype(int)


def _run_ablation(
        confidence_config,
        trajectories: list[np.ndarray],
        repeats: int,
        warmups: int,
        seed: int,
        *,
        batch_inputs: bool = True,
        reference_workers: int = 1,
        variant_indices: tuple[int, ...] | None = None,
) -> list[dict]:
    """Benchmark conventional Python against C++ variants."""
    if not trajectories:
        raise ValueError("Need at least one trajectory for ablation.")
    if reference_workers < 1 or (reference_workers > 1 and not batch_inputs):
        raise ValueError("Parallel reference requires batched C++ inputs.")
    selected = tuple(range(len(_ABLATION_CONFIGS))) if variant_indices is None else variant_indices
    if not selected or any(index not in range(len(_ABLATION_CONFIGS)) for index in selected):
        raise ValueError("Invalid benchmark variant indices")
    reference_settings = ConfidenceSequenceSettings(
        alpha=confidence_config.alpha,
        c=confidence_config.c,
        theta=confidence_config.theta,
        x_tolerance=confidence_config.x_tolerance,
        y_tolerance=confidence_config.y_tolerance,
        max_steps=confidence_config.max_steps,
        padding=confidence_config.padding,
    )
    builders = [
        lambda: ReferenceClassSelection(reference_settings),
        lambda: acs.ClassSelection(config=acs.ConfidenceSequenceSettings(
            alpha=confidence_config.alpha, c=confidence_config.c,
            theta=confidence_config.theta, x_tolerance=confidence_config.x_tolerance,
            y_tolerance=confidence_config.y_tolerance, max_steps=confidence_config.max_steps,
            padding=confidence_config.padding, endpoint_root_mode="full",
            root_finder="bisection"), stop_on_resolve=False),
        lambda: acs.ClassSelection(config=acs.ConfidenceSequenceSettings(
            alpha=confidence_config.alpha, c=confidence_config.c,
            theta=confidence_config.theta, x_tolerance=confidence_config.x_tolerance,
            y_tolerance=confidence_config.y_tolerance, max_steps=confidence_config.max_steps,
            padding=confidence_config.padding, endpoint_root_mode="componentwise",
            root_finder="bisection"), stop_on_resolve=False),
        lambda: acs.ClassSelection(config=acs.ConfidenceSequenceSettings(
            alpha=confidence_config.alpha, c=confidence_config.c,
            theta=confidence_config.theta, x_tolerance=confidence_config.x_tolerance,
            y_tolerance=confidence_config.y_tolerance, max_steps=confidence_config.max_steps,
            padding=confidence_config.padding, endpoint_root_mode="componentwise",
            root_finder="brent"), stop_on_resolve=False),
        lambda: acs.ClassSelection(config=acs.ConfidenceSequenceSettings(
            alpha=confidence_config.alpha, c=confidence_config.c,
            theta=confidence_config.theta, x_tolerance=confidence_config.x_tolerance,
            y_tolerance=confidence_config.y_tolerance, max_steps=confidence_config.max_steps,
            padding=confidence_config.padding, endpoint_root_mode="componentwise",
            root_finder="boost"), stop_on_resolve=False),
    ]

    available_cpus = _affinity()
    timings = [[] for _ in builders]
    order_rng = np.random.default_rng(seed + 60_000)
    pool_context = (
        ProcessPoolExecutor(
            max_workers=reference_workers,
            mp_context=get_context("spawn"),
            initializer=_restore_affinity,
            initargs=(available_cpus,),
        )
        if reference_workers > 1 and 0 in selected else nullcontext(None)
    )
    with pool_context as pool:
        if pool is not None:
            warmup_results = list(pool.map(
                _time_reference_task,
                ((reference_settings, draws) for draws in trajectories[:reference_workers]),
            ))
            worker_affinities = [count for _, count in warmup_results if count is not None]
            if worker_affinities and min(worker_affinities) < reference_workers:
                raise RuntimeError(
                    f"Python workers have only {min(worker_affinities)} CPU(s) in their affinity mask"
                )
            print(f"Python worker affinity: {worker_affinities}", flush=True)
        warmup_draws = trajectories[0]
        for variant_index in selected:
            builder = builders[variant_index]
            for _ in range(warmups):
                if variant_index == 0 or not batch_inputs:
                    _time_bounds(builder, warmup_draws)
                else:
                    _time_batch_bounds(builder, trajectories[:1])
        for repeat_index in range(repeats):
            for variant_index in order_rng.permutation(selected):
                run_seconds = 0.0
                builder = builders[int(variant_index)]
                if variant_index == 0 and pool is not None:
                    start = perf_counter()
                    worker_results = list(pool.map(
                        _time_reference_task,
                        ((reference_settings, draws) for draws in trajectories),
                    ))
                    if any(
                        count is not None and count < reference_workers
                        for _, count in worker_results
                    ):
                        raise RuntimeError("Python worker CPU affinity shrank during the run")
                    run_seconds = perf_counter() - start
                elif variant_index == 0 or not batch_inputs:
                    for draws in trajectories:
                        _, seconds = _time_bounds(builder, draws)
                        run_seconds += seconds
                else:
                    _, run_seconds = _time_batch_bounds(builder, trajectories)
                timings[int(variant_index)].append(run_seconds)
                print(
                    f"repeat {repeat_index + 1}/{repeats}: "
                    f"{_ABLATION_CONFIGS[int(variant_index)][0]} "
                    f"{run_seconds:.3f} s",
                    flush=True,
                )

    medians = {index: float(np.median(timings[index])) for index in selected}
    baseline = medians.get(0)
    updates_per_run = len(trajectories) * len(trajectories[0])
    rows = []
    for index in selected:
        config = _ABLATION_CONFIGS[index]
        values = timings[index]
        median = medians[index]
        slug, label, implementation, endpoint_mode, root_finder = config
        rows.append(
            {
                "id": slug,
                "label": label.replace("\n", " "),
                "implementation": implementation,
                "endpoint_root_mode": endpoint_mode,
                "root_finder": root_finder,
                "seconds": [float(value) for value in values],
                "median_seconds": median,
                "milliseconds_per_update": 1_000.0 * median / updates_per_run,
                "speedup_vs_reference": baseline / median if baseline is not None else None,
            }
        )
    return rows


def _plot_ablation(results: list[dict], output_path: Path) -> str:
    """Write a compact paper figure of aggregate speedups."""
    import matplotlib.pyplot as plt

    from bnn_demo_lib.plotting import COLORS, configure_matplotlib

    configure_matplotlib()
    by_id = {row["id"]: row for row in results}
    selected_ids = [
        "reference",
        "cpp-bisection",
        "cpp-bisection-componentwise",
        "cpp-boost-componentwise",
    ]
    selected = [by_id[slug] for slug in selected_ids]
    labels = [row["label"] for row in selected]
    colors = ("0.68", COLORS[0], COLORS[1], COLORS[2])
    speedups = [row["speedup_vs_reference"] for row in selected]
    largest = max(speedups)
    fig, axis = plt.subplots(1, 1, figsize=(4.8, 3.35))
    positions = np.arange(len(speedups))
    bars = axis.barh(positions, speedups, color=colors, edgecolor="white")
    axis.axvline(1.0, color="0.2", linewidth=1.0, linestyle="--", zorder=0)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.22 * largest)
    axis.bar_label(
        bars,
        labels=[f"{value:.2f}x" for value in speedups],
        padding=3,
        fontsize=10,
    )
    axis.set_xlabel("Speedup vs. reference")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return selected[-1]["root_finder"]


def _write_ablation_outputs(
    results: list[dict],
    output_json: Path,
    output_figure: Path,
    *,
    num_classes: int,
    max_samples: int,
    trajectories: int,
    easy_trajectories: int,
    hard_trajectories: int,
    repeats: int,
    warmups: int,
    seed: int,
    selected_indices: list[int],
    environment: dict[str, str | None] | None = None,
    execution_mode: str = "batched_inputs",
    reference_workers: int = 1,
) -> None:
    payload = {
        "schema_version": 3,
        "benchmark": "C++ bounding-box root and endpoint ablation",
        "num_classes": num_classes,
        "max_samples": max_samples,
        "trajectories": trajectories,
        "easy_trajectories": easy_trajectories,
        "hard_trajectories": hard_trajectories,
        "repeats": repeats,
        "warmups": warmups,
        "seed": seed,
        "input_source": "paired synthetic prediction streams from make_toy_data",
        "execution_mode": execution_mode,
        "reference_workers": reference_workers,
        "selected_indices": selected_indices,
        "environment": environment or {},
        "results": results,
    }
    selected_root_finder = _plot_ablation(results, output_figure)
    payload["figure_ids"] = [
        "reference",
        "cpp-bisection",
        "cpp-bisection-componentwise",
        f"cpp-{selected_root_finder}-componentwise",
    ]
    payload["selected_root_finder"] = selected_root_finder
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_figure}")
