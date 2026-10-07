"""Experiment-side configuration, sizes, and deterministic schedules."""

import math
from typing import Callable, Tuple

import adaptive_cs as acs
from omegaconf import DictConfig

CSBuilder = Callable[[], acs.ClassSelection | acs.DistributionAccuracy | acs.Threshold]

def config_from_omegaconf(cfg: DictConfig) -> acs.ConfidenceSequenceSettings:
    hedged = cfg.confidence
    return acs.ConfidenceSequenceSettings(
        alpha=float(hedged.alpha),
        c=float(hedged.c),
        theta=float(hedged.theta),
        x_tolerance=float(hedged.bisection_x_tolerance),
        y_tolerance=float(hedged.bisection_y_tolerance),
        max_steps=int(hedged.max_bisection_steps),
        padding=float(hedged.padding),
        simplex_intersection=bool(hedged.simplex_intersection),
        endpoint_root_mode=str(hedged.get("endpoint_root_mode", "componentwise")),
        root_finder=str(hedged.get("root_finder", "boost")),
    )


def geometric_inspection_times(
        max_samples: int,
        first: int = 16,
        growth: float = 1.5,
) -> Tuple[int, ...]:
    """Return predetermined geometrically spaced CS inspection times."""

    if max_samples < 1 or first < 1 or growth <= 1.0:
        raise ValueError("Require max_samples >= 1, first >= 1, and growth > 1.")
    times = []
    current = min(first, max_samples)
    while current < max_samples:
        times.append(current)
        current = max(current + 1, int(math.ceil(current * growth)))
    times.append(max_samples)
    return tuple(times)


def make_cs_builder(cfg: DictConfig) -> CSBuilder:
    config = config_from_omegaconf(cfg)
    return lambda: acs.ClassSelection(
        config=config, stop_on_resolve=False,
    )
