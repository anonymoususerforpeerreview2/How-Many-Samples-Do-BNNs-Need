"""Unit tests for the external adaptive allocation rules."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.toy import core as toy_core
from experiments.adaptive_baselines import (
    BaselineResult,
    bethell_variance_convergence,
    uncertrans_entropy_allocation,
)


def test_bethell_stops_after_required_consecutive_stable_variances() -> None:
    samples = np.full((20, 2, 3), (0.8, 0.1, 0.1), dtype=float)

    result = bethell_variance_convergence(samples, delta=0.0, patience=3)

    # The first comparison occurs at draw 2, so three stable comparisons end
    # at draw 4.  The returned mean is the mean at that stopping time.
    np.testing.assert_array_equal(result.stop_times, np.asarray([4, 4]))
    np.testing.assert_array_equal(result.quality_mask, np.asarray([True, True]))
    np.testing.assert_allclose(result.estimates, samples[0])


def test_bethell_marks_inputs_that_do_not_converge_by_the_cap() -> None:
    samples = np.asarray(
        [
            [[0.8, 0.2], [0.0, 1.0]],
            [[0.8, 0.2], [1.0, 0.0]],
            [[0.8, 0.2], [0.0, 1.0]],
            [[0.8, 0.2], [1.0, 0.0]],
        ],
        dtype=float,
    )

    result = bethell_variance_convergence(samples, delta=0.0, patience=2)

    np.testing.assert_array_equal(result.stop_times, np.asarray([3, 4]))
    np.testing.assert_array_equal(result.quality_mask, np.asarray([True, False]))
    assert result.quality_rate == 0.5


def test_uncertrans_clamps_entropy_rule_to_published_minimum_and_maximum() -> None:
    samples = np.empty((50, 2, 3), dtype=float)
    samples[:, 0] = (0.98, 0.01, 0.01)
    samples[:, 1] = (1 / 3, 1 / 3, 1 / 3)

    result = uncertrans_entropy_allocation(samples)

    np.testing.assert_array_equal(result.stop_times, np.asarray([10, 50]))
    np.testing.assert_array_equal(result.quality_mask, np.asarray([True, True]))


def test_toy_quality_metric_excludes_non_converged_baseline_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimates = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    baseline = BaselineResult(
        name="Bethell et al.",
        estimates=estimates,
        stop_times=np.asarray([3, 4]),
        quality_mask=np.asarray([True, False]),
    )
    monkeypatch.setattr(
        toy_core,
        "run_external_baselines",
        lambda *_args, **_kwargs: {"bethell": baseline},
    )
    data = SimpleNamespace(
        samples=np.repeat(estimates[None, :, :], 4, axis=0),
        labels=np.asarray([0, 0]),
        means=np.asarray([[0.8, 0.2], [0.8, 0.2]], dtype=float),
    )
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(max_samples=4, probability_threshold=0.7)
    )

    result = toy_core.run_baselines("class_selection", data, cfg)["bethell"]

    assert result.metric == pytest.approx(1.0)
    assert result.mean_samples == pytest.approx(3.5)
    assert result.quality_rate == pytest.approx(0.5)
