"""Synthetic predictive data used by the toy experiments."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class PredictiveDataset:
    """Parameters of a heterogeneous synthetic posterior-predictive dataset.

    ``component_means`` and ``component_weights`` describe a per-input mixture
    of Dirichlet distributions.  Ordinary Dirichlet inputs are represented by
    two identical components, which keeps downstream sampling uniform.
    """

    means: np.ndarray
    concentrations: np.ndarray
    labels: np.ndarray
    is_hard: np.ndarray
    component_means: np.ndarray
    component_weights: np.ndarray
    cohorts: np.ndarray


def sample_dirichlet_around(
    mean: np.ndarray,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one synthetic BNN softmax vector around a predictive mean."""
    return rng.dirichlet(np.maximum(1e-8, concentration * mean))


def sample_predictive_around(
    component_means: np.ndarray,
    component_weights: np.ndarray,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one prediction from an input-specific Dirichlet mixture."""
    component = int(rng.choice(len(component_weights), p=component_weights))
    return sample_dirichlet_around(component_means[component], concentration, rng)


def sample_predictive_streams(
    dataset: PredictiveDataset,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Generate paired posterior-predictive streams for every toy input."""
    draws = np.empty((max_samples, len(dataset.means), dataset.means.shape[1]))
    for index in range(len(dataset.means)):
        rng = np.random.default_rng(seed + index)
        for time in range(max_samples):
            draws[time, index] = sample_predictive_around(
                dataset.component_means[index],
                dataset.component_weights[index],
                dataset.concentrations[index],
                rng,
            )
    return draws


def _uniform_range(value: object, rng: np.random.Generator) -> float:
    lower, upper = map(float, value)
    return float(rng.uniform(lower, upper))


def _fill_remainder(
    mean: np.ndarray,
    classes: list[int],
    mass: float,
    rng: np.random.Generator,
) -> None:
    """Distribute positive residual mass without privileging a tail class."""
    if len(classes) == 1:
        mean[classes[0]] = mass
    else:
        mean[classes] = mass * rng.dirichlet(np.ones(len(classes)))


def make_predictive_dataset(cfg: DictConfig, seed: int) -> PredictiveDataset:
    """Create varied confidence and variance regimes for the toy experiment.

    The two-peak cohort has the same predictive mean as a single Dirichlet,
    but individual draws alternate between well-separated class-1/class-2
    modes.  It therefore tests stopping under posterior disagreement rather
    than only under diffuse Dirichlet noise.
    """
    num_inputs = int(cfg.num_inputs)
    num_classes = int(cfg.num_classes)
    if num_classes < 3:
        raise ValueError("The heterogeneous toy dataset requires at least 3 classes.")

    rng = np.random.default_rng(seed)
    cohort_cfgs = list(cfg.cohorts)
    fractions = np.asarray([float(cohort.fraction) for cohort in cohort_cfgs])
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("dataset.cohorts fractions must sum to one.")
    counts = np.floor(num_inputs * fractions).astype(int)
    counts[-1] += num_inputs - int(counts.sum())
    cohort_indices = np.repeat(np.arange(len(cohort_cfgs)), counts)
    rng.shuffle(cohort_indices)

    means = np.zeros((num_inputs, num_classes))
    concentrations = np.zeros(num_inputs)
    is_hard = np.zeros(num_inputs, dtype=bool)
    component_means = np.zeros((num_inputs, 2, num_classes))
    component_weights = np.full((num_inputs, 2), 0.5)
    cohorts = np.empty(num_inputs, dtype=object)

    for index, cohort_index in enumerate(cohort_indices):
        cohort = cohort_cfgs[int(cohort_index)]
        kind = str(cohort.kind)
        cohorts[index] = str(cohort.name)
        is_hard[index] = bool(cohort.hard)
        top_class = int(rng.integers(num_classes))
        other_classes = [c for c in range(num_classes) if c != top_class]
        rng.shuffle(other_classes)
        top = _uniform_range(cohort.top_probability, rng)
        mean = means[index]
        mean[top_class] = top

        if kind == "independent_remainder":
            _fill_remainder(mean, other_classes, 1.0 - top, rng)
        elif kind in {"near_tie", "two_peak"}:
            if kind == "near_tie":
                margin_lower, margin_upper = map(float, cohort.leader_margin)
                # Leave positive mass for the third and later classes while
                # retaining top as the unique leading probability.
                margin_lower = max(margin_lower, 2.0 * top - 1.0 + 1e-6)
                if margin_lower > margin_upper:
                    raise ValueError(
                        f"leader_margin is incompatible with top_probability for {cohort.name!r}."
                    )
                second = top - float(rng.uniform(margin_lower, margin_upper))
                remainder = 1.0 - top - second
            else:
                remainder = _uniform_range(cohort.tail_probability, rng)
                second = 1.0 - top - remainder
            if second <= 0.0 or remainder <= 0.0:
                raise ValueError(f"Invalid probabilities for cohort {cohort.name!r}.")
            mean[other_classes[0]] = second
            _fill_remainder(mean, other_classes[1:], remainder, rng)
        else:
            raise ValueError(f"Unknown toy cohort kind {kind!r}.")

        concentrations[index] = _uniform_range(cohort.kappa, rng)
        component_means[index] = mean
        if kind == "two_peak":
            shift = _uniform_range(cohort.mode_shift, rng)
            # Shift mass only between the two leading classes: each component
            # remains on the simplex and their equal-weight mixture is mean.
            if shift >= min(mean[top_class], mean[other_classes[0]]):
                raise ValueError(f"mode_shift is too large for cohort {cohort.name!r}.")
            component_means[index, 0, top_class] += shift
            component_means[index, 0, other_classes[0]] -= shift
            component_means[index, 1, top_class] -= shift
            component_means[index, 1, other_classes[0]] += shift

    labels = np.argmax(means, axis=1)
    return PredictiveDataset(
        means, concentrations, labels, is_hard, component_means, component_weights, cohorts
    )


def make_dataset(
    cfg: DictConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create predictive means, preserving the original public API.

    Cohort configurations use :func:`make_predictive_dataset`. The legacy
    fallback gives an ambiguous input the two leading configured masses and
    shares the remaining mass randomly among all other classes. With two
    classes, their relative configured weights are normalized instead. This
    reduces to the original construction when there are exactly three classes.
    """
    if "cohorts" in cfg:
        data = make_predictive_dataset(cfg, seed)
        return data.means, data.concentrations, data.labels, data.is_hard

    num_inputs = int(cfg.num_inputs)
    num_classes = int(cfg.num_classes)
    if num_classes < 2:
        raise ValueError("dataset.num_classes must be at least 2.")

    hard_top = float(cfg.hard.probabilities.top)
    hard_second = float(cfg.hard.probabilities.second)
    hard_remainder = 1.0 - hard_top - hard_second
    easy_top = float(cfg.easy.top_probability)
    probabilities = (hard_top, hard_second, hard_remainder, easy_top, 1.0 - easy_top)
    if min(probabilities) <= 0.0:
        raise ValueError("Configured class probabilities must be inside the simplex.")

    rng = np.random.default_rng(seed)
    means = np.zeros((num_inputs, num_classes))
    concentrations = np.zeros(num_inputs)
    is_hard = rng.random(num_inputs) < float(cfg.hard_fraction)

    for index, hard in enumerate(is_hard):
        top_class = int(rng.integers(0, num_classes))
        other_classes = [k for k in range(num_classes) if k != top_class]
        rng.shuffle(other_classes)

        if hard:
            if num_classes == 2:
                # With no third class to receive the configured remainder,
                # preserve the relative weight of the two ambiguous leaders.
                two_class_total = hard_top + hard_second
                means[index, top_class] = hard_top / two_class_total
                means[index, other_classes[0]] = hard_second / two_class_total
            else:
                means[index, top_class] = hard_top
                means[index, other_classes[0]] = hard_second
            if num_classes == 3:
                means[index, other_classes[1]] = hard_remainder
            elif num_classes > 3:
                remainder_weights = rng.dirichlet(np.ones(num_classes - 2))
                means[index, other_classes[1:]] = hard_remainder * remainder_weights
            kappa_cfg = cfg.hard.kappa
        else:
            means[index, top_class] = easy_top
            remainder_weights = rng.dirichlet(np.ones(num_classes - 1))
            means[index, other_classes] = (1.0 - easy_top) * remainder_weights
            kappa_cfg = cfg.easy.kappa

        concentrations[index] = rng.uniform(float(kappa_cfg.min), float(kappa_cfg.max))

    labels = np.argmax(means, axis=1)
    return means, concentrations, labels, is_hard


def select_example_indices(is_hard: np.ndarray) -> Tuple[int, int]:
    """Return the first ambiguous and first easy example indices."""
    hard = int(np.flatnonzero(is_hard)[0]) if np.any(is_hard) else 0
    easy = int(np.flatnonzero(~is_hard)[0]) if np.any(~is_hard) else 0
    return hard, easy
