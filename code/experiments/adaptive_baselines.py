"""External adaptive-Monte-Carlo baselines evaluated on a common draw stream.

These rules decide a per-input draw count only.  They deliberately do not
reimplement conformal prediction or UncerTrans's action-prediction model: the
comparison uses their sampling rules on the posterior-predictive draws used by
the confidence-sequence experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BETHELL_DELTA = 5e-4
BETHELL_PATIENCE = 10
UNCERTRANS_MIN_SAMPLES = 10
UNCERTRANS_MAX_SAMPLES = 50
UNCERTRANS_ENTROPY_MIN = 0.3
UNCERTRANS_ENTROPY_MAX = 0.7


@dataclass(frozen=True)
class BaselineResult:
    """Posterior means, allocations, and quality-eligible inputs for one rule."""

    name: str
    estimates: np.ndarray
    stop_times: np.ndarray
    quality_mask: np.ndarray

    @property
    def mean_samples(self) -> float:
        return float(self.stop_times.mean())

    @property
    def quality_rate(self) -> float:
        """Fraction of inputs eligible for the rule's quality metrics."""
        return float(self.quality_mask.mean())


def _final_means(samples: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(samples, axis=0, dtype=np.float64)
    indices = np.arange(samples.shape[1])
    return cumulative[stop_times - 1, indices] / stop_times[:, None]


def bethell_variance_convergence(
    samples: np.ndarray,
    *,
    delta: float = BETHELL_DELTA,
    patience: int = BETHELL_PATIENCE,
    max_samples: int | None = None,
) -> BaselineResult:
    """Algorithm 1 of Bethell et al. (2024), on precomputed predictions.

    The paper stops once the *sample-variance* vector changes by at most
    ``delta`` in every coordinate for ``patience`` consecutive forward passes.
    ``max_samples`` is the experiment's common cap; delta=.0005 and P=10 are
    the authors' public recommended/example settings.
    """
    if samples.ndim != 3:
        raise ValueError("samples must have shape (draws, inputs, classes)")
    total = min(len(samples), max_samples or len(samples))
    if total < 1 or patience < 1:
        raise ValueError("at least one sample and positive patience are required")

    values = samples[:total]
    n_inputs, n_classes = values.shape[1:]
    mean = values[0].astype(np.float64, copy=True)
    m2 = np.zeros((n_inputs, n_classes), dtype=np.float64)
    previous_variance = np.zeros_like(m2)
    consecutive = np.zeros(n_inputs, dtype=np.int64)
    stop_times = np.full(n_inputs, total, dtype=np.int64)
    active = np.ones(n_inputs, dtype=bool)
    converged = np.zeros(n_inputs, dtype=bool)

    for draw in range(2, total + 1):
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            break
        observation = values[draw - 1, active_indices]
        old_mean = mean[active_indices].copy()
        mean[active_indices] += (observation - old_mean) / draw
        m2[active_indices] += (observation - old_mean) * (
            observation - mean[active_indices]
        )
        variance = m2[active_indices] / draw
        stable = np.all(
            np.abs(variance - previous_variance[active_indices]) <= delta,
            axis=1,
        )
        consecutive[active_indices] = np.where(
            stable, consecutive[active_indices] + 1, 0
        )
        previous_variance[active_indices] = variance
        finished = active_indices[consecutive[active_indices] >= patience]
        stop_times[finished] = draw
        active[finished] = False
        converged[finished] = True

    return BaselineResult(
        "Bethell et al.",
        _final_means(values, stop_times),
        stop_times,
        converged,
    )


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, np.finfo(float).tiny, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=-1)


def uncertrans_entropy_allocation(
    samples: np.ndarray,
    *,
    min_samples: int = UNCERTRANS_MIN_SAMPLES,
    max_samples: int = UNCERTRANS_MAX_SAMPLES,
    entropy_min: float = UNCERTRANS_ENTROPY_MIN,
    entropy_max: float = UNCERTRANS_ENTROPY_MAX,
) -> BaselineResult:
    """Equation (22) of Zhai & Liu (2026) with the paper's Table 2 defaults."""
    if samples.ndim != 3:
        raise ValueError("samples must have shape (draws, inputs, classes)")
    if not 1 <= min_samples <= max_samples or entropy_max <= entropy_min:
        raise ValueError("invalid UncerTrans sampling settings")
    available = len(samples)
    initial_count = min(min_samples, available)
    initial_mean = samples[:initial_count].mean(axis=0)
    fraction = np.clip(
        (_entropy(initial_mean) - entropy_min) / (entropy_max - entropy_min), 0.0, 1.0
    )
    requested = min_samples + np.ceil(
        fraction * (max_samples - min_samples)
    ).astype(np.int64)
    stop_times = np.clip(requested, 1, available)
    return BaselineResult(
        "UncerTrans entropy",
        _final_means(samples, stop_times),
        stop_times,
        np.ones(samples.shape[1], dtype=bool),
    )


def run_external_baselines(
    samples: np.ndarray,
    *,
    max_samples: int | None = None,
) -> dict[str, BaselineResult]:
    """Run the two cited dynamic-allocation rules on a shared draw stream."""
    return {
        "bethell": bethell_variance_convergence(samples, max_samples=max_samples),
        "uncertrans": uncertrans_entropy_allocation(samples),
    }
