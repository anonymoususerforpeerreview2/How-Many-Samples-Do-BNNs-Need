"""Shared evaluation logic for the three CIFAR-100 stopping rules."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import adaptive_cs as acs
from experiments.adaptive_baselines import BaselineResult, run_external_baselines

from .config import Cifar100Config


TASKS = ("class_selection", "distribution_accuracy", "threshold")


@dataclass(frozen=True)
class TaskEvaluation:
    task: str
    fixed_metric: np.ndarray
    adaptive_metric: float
    certified_metric: float
    mean_samples: float
    certified_rate: float
    abstained_rate: float
    timeout_rate: float
    stop_times: np.ndarray
    statuses: np.ndarray
    estimates: np.ndarray
    decisions: np.ndarray | None
    # A fixed-budget comparator whose budget is the rounded all-input mean
    # number of draws consumed by the adaptive rule.  Saving the estimate (not
    # just its aggregate metric) lets plotting code compare paired error
    # distributions without re-running posterior inference.
    mean_matched_budget: int
    mean_matched_estimates: np.ndarray


@dataclass(frozen=True)
class BaselineEvaluation:
    """Quality and allocation of an external non-certified sampling rule."""

    name: str
    mean_samples: float
    quality_rate: float
    stop_times: np.ndarray
    estimates: np.ndarray
    quality_mask: np.ndarray
    metrics: dict[str, float]


def make_tracker(
    task: str, cfg: Cifar100Config, *, early_abstention: bool = True,
    inspection_times: tuple[int, ...] | None = None,
) -> acs.BoundingBoxTracker:
    # All three rules check for safe early abstention at inspections during
    # the final 30 observations before the cap.
    common = {
        "max_samples": cfg.task_max_samples(task),
        "config": cfg.confidence_settings(),
        "early_abstention": early_abstention,
        "inspection_times": inspection_times if inspection_times is not None else cfg.inspection_times_for(task),
    }
    if task == "class_selection":
        return acs.ClassSelection(**common)
    if task == "distribution_accuracy":
        return acs.DistributionAccuracy(epsilon=cfg.distribution_epsilon, **common)
    if task == "threshold":
        return acs.Threshold(threshold=cfg.probability_threshold, **common)
    raise ValueError(f"unknown task: {task}")


def run_tracker(
    task: str,
    samples: np.ndarray,
    cfg: Cifar100Config,
    *,
    early_abstention: bool = True,
    inspection_times: tuple[int, ...] | None = None,
) -> acs.CSResult:
    """Run the mandatory sparse-schedule tracker on an input batch."""
    tracker = make_tracker(task, cfg, early_abstention=early_abstention,
                           inspection_times=inspection_times)
    result = None
    while not tracker.done:
        active = tracker.active_indices
        if isinstance(active, slice):
            result = tracker.update(samples[tracker.time])
        else:
            result = tracker.update(samples[tracker.time, active], indices=active)
    assert result is not None
    return result


def _fixed_means(samples: np.ndarray) -> np.ndarray:
    counts = np.arange(1, len(samples) + 1, dtype=np.float64)[:, None, None]
    return np.cumsum(samples, axis=0, dtype=np.float64) / counts


def mean_matched_budget(mean_samples: float, max_samples: int) -> int:
    """Integer fixed budget nearest to an adaptive rule's mean allocation.

    A fixed-budget method necessarily uses an integer number of forward
    passes.  We use conventional half-up rounding and clip defensively to the
    available stream, while retaining the unrounded mean in summaries.
    """
    return int(np.clip(np.floor(mean_samples + 0.5), 1, max_samples))


def evaluate_task(
    task: str,
    samples: np.ndarray,
    reference_mean: np.ndarray,
    cfg: Cifar100Config,
) -> TaskEvaluation:
    samples = samples[: cfg.task_max_samples(task)]
    means = _fixed_means(samples)
    result = run_tracker(task, samples, cfg)
    statuses = np.asarray(result.status)
    certified = statuses == "stopped"
    estimates = result.mean.numpy()
    decisions = None if result.decision is None else result.decision.numpy()
    adaptive_mean_samples = float(result.n_samples.double().mean())
    matched_budget = mean_matched_budget(adaptive_mean_samples, len(samples))

    if task == "class_selection":
        target = reference_mean.argmax(axis=1)
        fixed_metric = np.mean(means.argmax(axis=2) == target[None, :], axis=1)
        adaptive_values = result.prediction.numpy() == target
    elif task == "distribution_accuracy":
        fixed_metric = np.mean(
            np.max(np.abs(means - reference_mean[None, :, :]), axis=2), axis=1
        )
        adaptive_values = np.max(np.abs(estimates - reference_mean), axis=1)
    elif task == "threshold":
        target = reference_mean.max(axis=1) >= cfg.probability_threshold
        fixed_metric = np.mean(
            (means.max(axis=2) >= cfg.probability_threshold) == target[None, :],
            axis=1,
        )
        assert decisions is not None
        adaptive_values = decisions == target
    else:
        raise ValueError(f"unknown task: {task}")

    # Adaptive prediction-quality metrics are conditional on certification.
    # Unresolved outputs remain available for diagnostics and their consumed
    # samples remain in mean_samples, but they carry no statistical guarantee
    # and therefore do not contribute to agreement or estimation error.
    adaptive_metric = (
        float(np.mean(adaptive_values[certified]))
        if np.any(certified)
        else float("nan")
    )
    # Retain the explicit field for compatibility with previously saved
    # summaries; it now names the same certified-only operating point.
    certified_metric = adaptive_metric
    return TaskEvaluation(
        task=task,
        fixed_metric=np.asarray(fixed_metric, dtype=float),
        adaptive_metric=adaptive_metric,
        certified_metric=certified_metric,
        mean_samples=adaptive_mean_samples,
        certified_rate=float(np.mean(certified)),
        abstained_rate=float(np.mean(statuses == "abstained")),
        timeout_rate=float(np.mean(statuses == "timeout")),
        stop_times=result.n_samples.numpy(),
        statuses=statuses,
        estimates=estimates,
        decisions=decisions,
        mean_matched_budget=matched_budget,
        mean_matched_estimates=means[matched_budget - 1].copy(),
    )


def evaluate_external_baselines(
    samples: np.ndarray,
    reference_mean: np.ndarray,
    cfg: Cifar100Config,
) -> dict[str, BaselineEvaluation]:
    """Apply both external allocation rules to the shared posterior stream.

    Bethell outputs contribute to quality only if its variance-stability rule
    fires by the common evaluation horizon. Capped, non-converged inputs still
    contribute their full cost to mean allocation. UncerTrans retains its
    published 10--50 defaults and has no convergence condition.
    """
    raw: dict[str, BaselineResult] = run_external_baselines(
        samples[: cfg.evaluation_samples], max_samples=cfg.evaluation_samples
    )
    class_target = reference_mean.argmax(axis=1)
    threshold_target = reference_mean.max(axis=1) >= cfg.probability_threshold
    output: dict[str, BaselineEvaluation] = {}
    for key, result in raw.items():
        estimate = result.estimates
        quality_mask = result.quality_mask
        if np.any(quality_mask):
            class_metric = float(
                np.mean((estimate.argmax(axis=1) == class_target)[quality_mask])
            )
            distribution_metric = float(
                np.mean(
                    np.max(np.abs(estimate - reference_mean), axis=1)[quality_mask]
                )
            )
            threshold_metric = float(
                np.mean(
                    (
                        (estimate.max(axis=1) >= cfg.probability_threshold)
                        == threshold_target
                    )[quality_mask]
                )
            )
        else:
            class_metric = distribution_metric = threshold_metric = float("nan")
        output[key] = BaselineEvaluation(
            name=result.name,
            mean_samples=result.mean_samples,
            quality_rate=result.quality_rate,
            stop_times=result.stop_times,
            estimates=estimate,
            quality_mask=quality_mask,
            metrics={
                "class_selection": class_metric,
                "distribution_accuracy": distribution_metric,
                "threshold": threshold_metric,
            },
        )
    return output


def matched_fixed_budget(fixed_metric: np.ndarray, adaptive_metric: float, task: str) -> int:
    if not np.isfinite(adaptive_metric):
        return len(fixed_metric)
    matches = (
        np.flatnonzero(fixed_metric <= adaptive_metric)
        if task == "distribution_accuracy"
        else np.flatnonzero(fixed_metric >= adaptive_metric)
    )
    return int(matches[0] + 1) if matches.size else len(fixed_metric)


def task_summary(result: TaskEvaluation, cfg: Cifar100Config) -> dict[str, Any]:
    return {
        "inspection_times": cfg.inspection_times_for(result.task),
        "adaptive_metric": result.adaptive_metric,
        "certified_metric": result.certified_metric,
        "adaptive_mean_samples": result.mean_samples,
        "adaptive_certified_rate": result.certified_rate,
        "adaptive_abstained_rate": result.abstained_rate,
        "adaptive_timeout_rate": result.timeout_rate,
        "adaptive_median_samples": float(np.median(result.stop_times)),
        "adaptive_q25_samples": float(np.quantile(result.stop_times, 0.25)),
        "adaptive_q75_samples": float(np.quantile(result.stop_times, 0.75)),
        "mean_matched_fixed_budget": result.mean_matched_budget,
        "matched_fixed_budget": matched_fixed_budget(
            result.fixed_metric, result.adaptive_metric, result.task
        ),
    }
