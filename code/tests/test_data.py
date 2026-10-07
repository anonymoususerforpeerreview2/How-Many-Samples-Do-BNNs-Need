"""Checks for configurable synthetic classification data."""

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn_demo_lib.data import make_dataset, make_predictive_dataset  # noqa: E402


def _config(num_classes: int):
    cfg = OmegaConf.load(ROOT / "experiments" / "toy" / "config.yaml").dataset
    cfg.num_classes = num_classes
    return cfg


def test_multiclass_dataset_stays_on_simplex() -> None:
    means, _, labels, is_hard = make_dataset(_config(7), seed=3)

    assert means.shape == (500, 7)
    assert np.all(means > 0.0)
    assert np.allclose(means.sum(axis=1), 1.0)
    assert np.array_equal(labels, means.argmax(axis=1))
    hard_means = means[is_hard]
    assert len(hard_means) == 250
    assert np.all(np.max(hard_means, axis=1) < 0.70)


def test_two_peak_cohort_has_separated_simplex_modes() -> None:
    data = make_predictive_dataset(_config(3), seed=3)
    multi = data.cohorts == "multimodal_disagreement"

    assert multi.sum() == 125
    assert np.allclose(data.component_means[multi].mean(axis=1), data.means[multi])
    assert np.all(np.max(np.abs(np.diff(data.component_means[multi], axis=1)), axis=(1, 2)) > 0.30)
