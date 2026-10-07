"""Canonical data generation and evaluation for the paper's toy experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

import adaptive_cs as acs
from experiments.adaptive_baselines import BaselineResult, run_external_baselines
from bnn_demo_lib.data import (
    make_predictive_dataset,
    sample_predictive_streams,
    select_example_indices,
)
from bnn_demo_lib.settings import config_from_omegaconf


CONFIG_PATH = Path(__file__).with_name("config.yaml")
TASK_LABELS = {
    "class_selection": "Class selection",
    "distribution_accuracy": "Distribution accuracy",
    "threshold": "Threshold decision",
}


@dataclass(frozen=True)
class ToyData:
    means: np.ndarray
    concentrations: np.ndarray
    labels: np.ndarray
    is_hard: np.ndarray
    samples: np.ndarray
    component_means: np.ndarray
    component_weights: np.ndarray
    cohorts: np.ndarray


@dataclass(frozen=True)
class ToyTaskResult:
    task: str
    fixed_budgets: np.ndarray
    fixed_metric: np.ndarray
    adaptive_mean_samples: float
    adaptive_metric: float
    adaptive_all_input_metric: float
    adaptive_unresolved_metric: float
    adaptive_certified_rate: float
    adaptive_abstained_rate: float
    adaptive_timeout_rate: float
    stop_times: np.ndarray
    statuses: np.ndarray
    estimates: np.ndarray
    mean_matched_budget: int
    mean_matched_estimates: np.ndarray


@dataclass(frozen=True)
class ToyBaselineResult:
    """One external rule's operating point for a toy evaluation task."""

    name: str
    mean_samples: float
    metric: float
    quality_rate: float
    stop_times: np.ndarray
    estimates: np.ndarray
    quality_mask: np.ndarray


def load_config() -> DictConfig:
    """Load the single checked-in configuration used by every toy figure."""
    return OmegaConf.load(CONFIG_PATH)


def make_toy_data(cfg: DictConfig | None = None) -> ToyData:
    """Generate the dataset and one paired posterior-draw stream."""
    cfg = cfg or load_config()
    seed = int(cfg.experiment.seed)
    max_samples = int(cfg.experiment.max_samples)
    dataset = make_predictive_dataset(cfg.dataset, seed)
    samples = sample_predictive_streams(dataset, max_samples, seed + 10_000)
    return ToyData(
        dataset.means,
        dataset.concentrations,
        dataset.labels,
        dataset.is_hard,
        samples,
        dataset.component_means,
        dataset.component_weights,
        dataset.cohorts,
    )


def _tracker(task: str, cfg: DictConfig) -> acs.BoundingBoxTracker:
    common = {
        "max_samples": int(cfg.experiment.max_samples),
        "config": config_from_omegaconf(cfg),
        "early_abstention": bool(cfg.experiment.early_abstention),
    }
    if task == "class_selection":
        return acs.ClassSelection(**common)
    if task == "distribution_accuracy":
        return acs.DistributionAccuracy(
            epsilon=float(cfg.experiment.distribution_epsilon), **common
        )
    if task == "threshold":
        return acs.Threshold(
            threshold=float(cfg.experiment.probability_threshold), **common
        )
    raise ValueError(f"unknown task: {task}")


def _run_tracker(task: str, data: ToyData, cfg: DictConfig) -> acs.CSResult:
    tracker = _tracker(task, cfg)
    result = None
    while not tracker.done:
        active = tracker.active_indices
        if isinstance(active, slice):
            result = tracker.update(data.samples[tracker.time])
        else:
            result = tracker.update(data.samples[tracker.time, active], indices=active)
    assert result is not None
    return result


def _mean_matched_budget(mean_samples: float, max_samples: int) -> int:
    """Return the integer fixed budget closest to the adaptive mean count."""
    return int(np.clip(np.floor(mean_samples + 0.5), 1, max_samples))


def run_task(
    task: str,
    data: ToyData | None = None,
    cfg: DictConfig | None = None,
) -> ToyTaskResult:
    """Evaluate one stopping task against paired fixed-budget estimates."""
    cfg = cfg or load_config()
    data = data or make_toy_data(cfg)
    budgets = np.asarray(cfg.experiment.fixed_budgets, dtype=int)
    cumulative_mean = np.cumsum(data.samples, axis=0)
    fixed_means = cumulative_mean[budgets - 1] / budgets[:, None, None]
    result = _run_tracker(task, data, cfg)
    estimates = result.mean.numpy()
    statuses = np.asarray(result.status)
    certified = statuses == "stopped"

    if task == "class_selection":
        fixed_metric = np.mean(
            fixed_means.argmax(axis=2) == data.labels[None, :], axis=1
        )
        adaptive_values = result.prediction.numpy() == data.labels
        all_input_values = adaptive_values
    elif task == "distribution_accuracy":
        fixed_metric = np.mean(
            np.max(np.abs(fixed_means - data.means[None, :, :]), axis=2), axis=1
        )
        adaptive_values = np.max(np.abs(estimates - data.means), axis=1)
        all_input_values = adaptive_values
    else:
        threshold = float(cfg.experiment.probability_threshold)
        targets = data.means.max(axis=1) >= threshold
        fixed_metric = np.mean(
            (fixed_means.max(axis=2) >= threshold) == targets[None, :], axis=1
        )
        assert result.decision is not None
        adaptive_values = result.decision.numpy() == targets
        all_input_values = (estimates.max(axis=1) >= threshold) == targets

    adaptive_metric = (
        float(np.mean(adaptive_values[certified])) if np.any(certified) else float("nan")
    )
    # Score final point estimates for every input, including those for which
    # the confidence sequence did not certify a decision.
    adaptive_all_input_metric = float(np.mean(all_input_values))
    adaptive_unresolved_metric = (
        float(np.mean(all_input_values[~certified]))
        if np.any(~certified)
        else float("nan")
    )
    adaptive_mean_samples = float(result.n_samples.double().mean())
    matched_budget = _mean_matched_budget(adaptive_mean_samples, len(data.samples))
    return ToyTaskResult(
        task=task,
        fixed_budgets=budgets,
        fixed_metric=np.asarray(fixed_metric, dtype=float),
        adaptive_mean_samples=adaptive_mean_samples,
        adaptive_metric=adaptive_metric,
        adaptive_all_input_metric=adaptive_all_input_metric,
        adaptive_unresolved_metric=adaptive_unresolved_metric,
        adaptive_certified_rate=float(np.mean(certified)),
        adaptive_abstained_rate=float(np.mean(statuses == "abstained")),
        adaptive_timeout_rate=float(np.mean(statuses == "timeout")),
        stop_times=result.n_samples.numpy(),
        statuses=statuses,
        estimates=estimates,
        mean_matched_budget=matched_budget,
        mean_matched_estimates=cumulative_mean[matched_budget - 1] / matched_budget,
    )


def run_baselines(
    task: str,
    data: ToyData | None = None,
    cfg: DictConfig | None = None,
) -> dict[str, ToyBaselineResult]:
    """Evaluate the two external sample-allocation rules for one task.

    Bethell estimates contribute to quality only when its variance-stability
    rule fires by the cap. UncerTrans has no convergence condition and always
    contributes because it directly assigns a finite sample count.
    """
    cfg = cfg or load_config()
    data = data or make_toy_data(cfg)
    raw: dict[str, BaselineResult] = run_external_baselines(
        data.samples, max_samples=int(cfg.experiment.max_samples)
    )
    if task == "class_selection":
        values = lambda estimate: estimate.argmax(axis=1) == data.labels
    elif task == "distribution_accuracy":
        values = lambda estimate: np.max(np.abs(estimate - data.means), axis=1)
    elif task == "threshold":
        threshold = float(cfg.experiment.probability_threshold)
        targets = data.means.max(axis=1) >= threshold
        values = lambda estimate: (estimate.max(axis=1) >= threshold) == targets
    else:
        raise ValueError(f"unknown task: {task}")
    return {
        key: ToyBaselineResult(
            name=result.name,
            mean_samples=result.mean_samples,
            metric=(
                float(np.mean(values(result.estimates)[result.quality_mask]))
                if np.any(result.quality_mask)
                else float("nan")
            ),
            quality_rate=result.quality_rate,
            stop_times=result.stop_times,
            estimates=result.estimates,
            quality_mask=result.quality_mask,
        )
        for key, result in raw.items()
    }


def _first_certified_time(
    samples: np.ndarray,
    tracker_type: type[acs.BoundingBoxTracker],
    **kwargs: Any,
) -> int | None:
    tracker = tracker_type(
        max_samples=len(samples),
        stop_on_resolve=False,
        **kwargs,
    )
    stop = None
    for sample_count, sample in enumerate(samples, start=1):
        result = tracker.update(sample)
        if stop is None and bool(result.certified[0]):
            stop = sample_count
    return stop


def paper_example_stopping_times(
    data: ToyData | None = None,
    cfg: DictConfig | None = None,
    example_indices: tuple[int, int] | None = None,
) -> dict[str, dict[str, int | None]]:
    """Compute the three rule stopping times printed in the example caption."""
    cfg = cfg or load_config()
    data = data or make_toy_data(cfg)
    settings = config_from_omegaconf(cfg)
    hard, easy = example_indices or paper_example_indices(data)
    output: dict[str, dict[str, int | None]] = {}
    for difficulty, index in (
        ("hard", hard),
        ("easy", easy),
    ):
        common = {"samples": data.samples[:, index], "config": settings}
        output[difficulty] = {
            "class": _first_certified_time(
                tracker_type=acs.ClassSelection, **common
            ),
            "distribution": _first_certified_time(
                tracker_type=acs.DistributionAccuracy,
                epsilon=float(cfg.experiment.distribution_epsilon),
                **common,
            ),
            "threshold": _first_certified_time(
                tracker_type=acs.Threshold,
                threshold=float(cfg.experiment.probability_threshold),
                **common,
            ),
        }
    return output


def paper_example_indices(data: ToyData) -> tuple[int, int]:
    """Select stable and moderately ambiguous examples for the paper figure."""
    easy_candidates = np.flatnonzero(data.cohorts == "confident_stable")
    hard_candidates = np.flatnonzero(data.cohorts == "multimodal_disagreement")
    if len(easy_candidates) and len(hard_candidates):
        # Keep the example resolvable while making the two leading classes
        # close enough that class certification is visibly non-immediate.
        hard_means = data.means[hard_candidates]
        ordered = np.sort(hard_means, axis=1)
        leader_margins = ordered[:, -1] - ordered[:, -2]
        target_margin = 0.22
        hard = hard_candidates[np.argmin(np.abs(leader_margins - target_margin))]
        return int(hard), int(easy_candidates[0])
    return select_example_indices(data.is_hard)


def write_tex_macros(path: Path, macros: dict[str, str]) -> None:
    """Write deterministic LaTeX commands generated from experiment results."""
    lines = ["% Generated by the adjacent figure script; do not edit manually."]
    lines.extend(
        f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
