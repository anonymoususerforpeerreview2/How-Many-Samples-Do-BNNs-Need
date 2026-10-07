"""Behavioral checks for the native-backed public tracker API."""

import numpy as np
import pytest
import torch

import adaptive_cs as cs
from adaptive_cs._cpp import CppRootTracer
from non_vectorized_class_selection import (
    ClassSelection as ReferenceConfidenceSequence,
    ConfidenceSequenceSettings as ReferenceSettings,
    unconstrained_bet,
)


def test_confidence_sequence_settings_replaces_bisection_config_alias():
    assert hasattr(cs, "ConfidenceSequenceSettings")
    assert not hasattr(cs, "BisectionConfig")
    assert not hasattr(cs, "BisectionSettings")
    assert "BisectionConfig" not in cs.__all__
    assert "BisectionSettings" not in cs.__all__


def test_paper_examples_use_the_public_api():
    single = cs.ClassSelection(alpha=0.05, max_samples=200)
    while not single.done:
        single.update(np.asarray([0.99, 0.01]))
    assert single.prediction == 0
    assert single.mean.shape == (2,)
    assert single.lower.shape == (2,)
    assert single.upper.shape == (2,)
    assert not hasattr(single, "active")

    batch = cs.DistributionAccuracy(epsilon=0.02, alpha=0.05, max_samples=3)
    inputs = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    assert batch.active_indices == slice(None)
    while not batch.done:
        indices = batch.active_indices
        selected = inputs[indices]
        values = (
            np.stack([selected, selected])
            if batch.time == 0
            else selected
        )
        batch.update(values, indices=indices)
    assert batch.prediction.tolist() == [0, 1]
    assert batch.active_indices.size == 0
    assert batch.n_samples.tolist() == [3, 3]


def test_numpy_and_torch_inputs_and_final_chunk_bounds():
    tracker = cs.ClassSelection(max_samples=5)
    result = tracker.update(np.asarray([[[0.8, 0.2]], [[0.7, 0.3]]]))
    assert result.mean.shape == (1, 2)
    assert result.n_samples.tolist() == [2]
    next_result = tracker.update(torch.tensor([[0.6, 0.4]]))
    assert next_result.n_samples.tolist() == [3]
    assert tracker.remaining == 2


def test_scheduled_inspections_consume_every_draw_but_only_stop_on_schedule():
    schedule = (8, 16, 24)
    draws = np.repeat(np.asarray([[[0.99, 0.01]]]), 24, axis=0)
    tracker = cs.ClassSelection(max_samples=24, inspection_times=schedule)
    result = tracker.update(draws)

    assert result.n_samples[0].item() in schedule
    assert result.n_samples[0].item() >= 8
    assert result.status[0] in {"stopped", "timeout"}

    one_at_a_time = cs.ClassSelection(max_samples=24, inspection_times=schedule)
    other = None
    while not one_at_a_time.done:
        other = one_at_a_time.update([0.99, 0.01])
    assert other is not None
    assert other.n_samples.tolist() == result.n_samples.tolist()
    assert other.status == result.status


def test_inspection_schedule_validates_and_includes_cap():
    tracker = cs.ClassSelection(max_samples=9, inspection_times=(3, 6))
    assert tracker.inspection_times == (3, 6, 9)
    with pytest.raises(ValueError):
        cs.ClassSelection(inspection_times=(1, 2))
    with pytest.raises(ValueError):
        cs.ClassSelection(max_samples=5, inspection_times=(0, 3))
    with pytest.raises(ValueError):
        cs.ClassSelection(max_samples=5, inspection_times=(3, 6))


def test_active_batch_shrinks_and_sample_counts_diverge():
    tracker = cs.ClassSelection(max_samples=30)
    easy = np.asarray([0.99, 0.01])
    hard = np.asarray([0.5, 0.5])
    result = tracker.update(np.stack([easy, hard]))
    while not tracker.done and tracker.active_indices.size == 2:
        result = tracker.update(np.stack([easy, hard]), indices=tracker.active_indices)
    if tracker.active_indices.size == 1:
        active = tracker.active_indices
        before = tracker.n_samples.tolist()
        repeated = np.repeat(hard[None, None, :], 2, axis=0)
        result = tracker.update(repeated, indices=active)
        assert result.n_samples[0].item() == before[0]
        assert result.n_samples[1].item() == before[1] + 2
        assert len(set(result.n_samples.tolist())) > 1
    assert result.status[0] in {"running", "stopped", "timeout", "empty"}


@pytest.mark.parametrize(
    "tracker_type, kwargs",
    [
        (cs.DistributionAccuracy, {"epsilon": 0.2}),
        (cs.Threshold, {"threshold": 0.7}),
    ],
)
def test_remaining_probability_trackers(tracker_type, kwargs):
    tracker = tracker_type(max_samples=5, **kwargs)
    result = tracker.update(torch.tensor([0.9, 0.1]))
    assert result.lower.shape == (1, 2)
    assert result.decision is None or result.decision.shape == (1,)


def test_threshold_decisions_persist_after_an_input_leaves_the_active_batch():
    tracker = cs.Threshold(max_samples=200, threshold=0.51)
    easy = np.asarray([1.0, 0.0])
    hard = np.asarray([0.5, 0.5])
    result = tracker.update(np.stack([easy, hard]))
    while not tracker.done and tracker.active_indices.size == 2:
        result = tracker.update(np.stack([easy, hard]), indices=tracker.active_indices)
    assert result.decision is not None
    assert bool(result.decision[0])
    if tracker.active_indices.size == 1:
        result = tracker.update(np.asarray([[0.5, 0.5]]), indices=tracker.active_indices)
        assert bool(result.decision[0])


def test_threshold_positive_decision_uses_empirical_leader():
    tracker = cs.Threshold(max_samples=10, threshold=0.3)
    tracker.update(np.full((2, 3), 1.0 / 3.0))
    tracker._sum[:] = [[0.40, 0.35, 0.25], [0.25, 0.45, 0.30]]
    tracker._lower[:] = [[0.20, 0.31, 0.10], [0.10, 0.31, 0.15]]
    tracker._upper[:] = [[0.50, 0.40, 0.35], [0.40, 0.60, 0.40]]

    newly = tracker._check_stopping(np.arange(2))

    assert newly.tolist() == [False, True]
    assert tracker._state.tolist() == [0, 1]
    assert tracker._last_decision.tolist() == [False, True]


def test_uncertainty_accuracy_and_threshold_modes():
    accuracy = cs.Uncertainty(
        component="aleatoric", stopping="accuracy", epsilon=0.5, max_samples=3
    )
    threshold = cs.Uncertainty(
        component="epistemic", stopping="threshold", threshold=0.2, max_samples=3
    )
    draw = np.asarray([0.9, 0.1])
    accuracy_result = accuracy.update(draw)
    threshold_result = threshold.update(draw)
    assert accuracy_result.total is not None
    assert accuracy_result.aleatoric is not None
    assert accuracy_result.epistemic is not None
    assert threshold_result.decision.shape == (1,)

    with pytest.raises(ValueError):
        cs.Uncertainty(component="total", stopping="interval_width", epsilon=0.1)


def test_early_abstention_reports_abstained():
    tracker = cs.ClassSelection(max_samples=8, early_abstention=True)
    draw = np.full((1, 3), 1.0 / 3.0)
    while not tracker.done:
        active = tracker.active_indices
        tracker.update(draw if isinstance(active, slice) else draw[active],
                        indices=None if isinstance(active, slice) else active)
    assert tracker.resolved
    assert tracker._result(np.zeros(1, dtype=bool)).status[0] in {"abstained", "timeout", "empty"}


@pytest.mark.parametrize(
    "tracker_type, kwargs",
    [
        (cs.ClassSelection, {}),
        (cs.DistributionAccuracy, {"epsilon": 0.05}),
        (cs.Threshold, {"threshold": 1.0 / 3.0}),
    ],
)
def test_futility_is_checked_only_in_the_final_30_draws(
    tracker_type, kwargs, monkeypatch
):
    tracker = tracker_type(
        max_samples=40,
        inspection_times=(9, 10, 40),
        early_abstention=True,
        **kwargs,
    )
    calls = []
    bound_times = []
    draw = np.full((1, 3), 1.0 / 3.0)
    tracker.update(draw)
    native_update = tracker._cpp.update
    def record_update(*args, **kwargs):
        if kwargs["compute_bounds"]:
            bound_times.append(tracker.time + 1)
        return native_update(*args, **kwargs)
    monkeypatch.setattr(tracker._cpp, "update", record_update)
    monkeypatch.setattr(
        tracker._cpp,
        "futile",
        lambda *args: calls.append(tracker.time) or np.asarray([False]),
    )
    for _ in range(8):
        tracker.update(draw)
    assert calls == []
    tracker.update(draw)
    assert calls == [10]
    for _ in range(30):
        tracker.update(draw)
    assert calls == [10, 20, 30, 35]
    assert bound_times == [9, 10, 40]


def test_root_finder_and_simplex_settings_are_accepted():
    for root_finder in ("brent", "bisection", "boost"):
        tracker = cs.ClassSelection(
            config=cs.ConfidenceSequenceSettings(
                root_finder=root_finder,
                endpoint_root_mode="componentwise",
                simplex_intersection=True,
            ),
            stop_on_resolve=False,
        )
        for _ in range(20):
            result = tracker.update([0.6, 0.3, 0.1])
        assert np.all(result.lower.numpy() <= result.upper.numpy())


def test_toms748_diagnostic_records_nested_brackets_and_evaluations():
    observations = np.asarray([0.2, 0.4, 0.3, 0.5], dtype=float)
    bets = []
    running_sum = 0.0
    sum_deviation = 0.0
    variance = 0.25
    for time, observation in enumerate(observations, start=1):
        bets.append(float(unconstrained_bet(0.05, variance * time, time)))
        regularized_mean = (0.5 + running_sum + observation) / (time + 1.0)
        sum_deviation += (observation - regularized_mean) ** 2
        running_sum += observation
        variance = (0.25 + sum_deviation) / (time + 1.0)

    trace = CppRootTracer().trace(observations, np.asarray(bets), "crossing")
    assert trace["status"] == 0
    assert trace["function_evaluations"].tolist() == list(
        range(2, trace["total_function_evaluations"] + 1)
    )
    assert np.all(np.diff(trace["lower"]) >= 0.0)
    assert np.all(np.diff(trace["upper"]) <= 0.0)
    assert trace["root_lower"] <= trace["root_upper"]


def test_confidence_boxes_are_intersected_over_time():
    """Returned boxes must tighten even when freshly inverted boxes do not."""
    rng = np.random.default_rng(29)
    draws = rng.dirichlet((3.0, 2.0, 1.0), size=40)
    tracker = cs.ClassSelection(max_samples=40, stop_on_resolve=False)
    previous_lower = np.zeros(3)
    previous_upper = np.ones(3)

    for draw in draws:
        result = tracker.update(draw)
        lower, upper = result.lower.numpy()[0], result.upper.numpy()[0]
        assert np.all(lower >= previous_lower - 1e-12)
        assert np.all(upper <= previous_upper + 1e-12)
        previous_lower, previous_upper = lower, upper


def test_high_dimensional_bounds_match_python_reference():
    """Guard the multivariate inversion used by distribution stopping."""
    rng = np.random.default_rng(14)
    draws = rng.dirichlet(np.linspace(0.5, 2.0, 8), size=18)
    reference_settings = ReferenceSettings()
    reference = ReferenceConfidenceSequence(reference_settings)
    native = cs.DistributionAccuracy(
        epsilon=0.05,
        stop_on_resolve=False,
        config=cs.ConfidenceSequenceSettings(
            endpoint_root_mode="full",
            root_finder="bisection",
            simplex_intersection=False,
        ),
    )
    for draw in draws:
        expected = reference.update(draw)
        actual = native.update(draw)
    np.testing.assert_allclose(actual.lower.numpy()[0], expected.lower, atol=2e-4)
    np.testing.assert_allclose(actual.upper.numpy()[0], expected.upper, atol=2e-4)
