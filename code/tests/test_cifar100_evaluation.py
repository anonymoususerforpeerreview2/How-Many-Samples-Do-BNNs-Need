"""Metric-policy checks for adaptive CIFAR-100 evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.adaptive_baselines import BaselineResult
from experiments.cifar100 import evaluation
from experiments.cifar100.config import Cifar100Config


def test_all_cifar_trackers_receive_the_sparse_inspection_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict] = []

    def tracker_factory(**kwargs):
        received.append(kwargs)
        return object()

    monkeypatch.setattr(evaluation.acs, "ClassSelection", tracker_factory)
    monkeypatch.setattr(evaluation.acs, "DistributionAccuracy", tracker_factory)
    monkeypatch.setattr(evaluation.acs, "Threshold", tracker_factory)
    cfg = Cifar100Config(run_dir="run", data_root="data")

    for task in evaluation.TASKS:
        evaluation.make_tracker(task, cfg)

    assert [call["inspection_times"] for call in received] == [
        cfg.sparse_inspections,
        cfg.sparse_inspections,
        cfg.sparse_inspections,
    ]
    assert all(call["early_abstention"] for call in received)
    evaluation.make_tracker("class_selection", cfg, early_abstention=False)
    assert received[-1]["early_abstention"] is False


@pytest.mark.parametrize(
    "task, expected_metric",
    [
        ("class_selection", 1.0),
        ("distribution_accuracy", 0.05),
        ("threshold", 1.0),
    ],
)
def test_quality_uses_certified_inputs_but_sample_cost_uses_all_inputs(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    expected_metric: float,
) -> None:
    reference_mean = np.asarray(
        [[0.90, 0.10], [0.20, 0.80], [0.55, 0.45]], dtype=float
    )
    estimates = torch.tensor(
        [[0.85, 0.15], [0.90, 0.10], [0.10, 0.90]], dtype=torch.float64
    )
    fake_result = SimpleNamespace(
        status=["stopped", "timeout", "abstained"],
        mean=estimates,
        prediction=torch.tensor([0, 0, 1]),
        decision=(
            torch.tensor([True, False, True]) if task == "threshold" else None
        ),
        n_samples=torch.tensor([1, 2, 1]),
    )
    monkeypatch.setattr(evaluation, "run_tracker", lambda *_args, **_kwargs: fake_result)

    samples = np.asarray(
        [
            [[0.8, 0.2], [0.8, 0.2], [0.2, 0.8]],
            [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]],
        ],
        dtype=float,
    )
    cfg = SimpleNamespace(
        probability_threshold=0.7,
        task_max_samples=lambda _task: 2,
    )

    result = evaluation.evaluate_task(task, samples, reference_mean, cfg)

    assert result.adaptive_metric == pytest.approx(expected_metric)
    assert result.certified_metric == pytest.approx(expected_metric)
    assert result.certified_rate == pytest.approx(1 / 3)
    assert result.mean_samples == pytest.approx(4 / 3)
    assert result.mean_matched_budget == 1
    np.testing.assert_allclose(result.mean_matched_estimates, samples[0])


def test_external_baselines_emit_all_input_metrics_and_allocations() -> None:
    samples = np.asarray(
        [
            [[0.9, 0.1], [0.5, 0.5]],
        ]
        * 20,
        dtype=float,
    )
    reference_mean = samples[0]
    cfg = SimpleNamespace(evaluation_samples=20, probability_threshold=0.7)

    results = evaluation.evaluate_external_baselines(samples, reference_mean, cfg)

    assert set(results) == {"bethell", "uncertrans"}
    for result in results.values():
        assert result.estimates.shape == reference_mean.shape
        assert np.all((1 <= result.stop_times) & (result.stop_times <= 20))
        assert result.quality_rate == 1.0
        assert set(result.metrics) == {
            "class_selection",
            "distribution_accuracy",
            "threshold",
        }


def test_external_baseline_quality_excludes_non_converged_inputs(
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
        evaluation,
        "run_external_baselines",
        lambda *_args, **_kwargs: {"bethell": baseline},
    )
    samples = np.repeat(estimates[None, :, :], 4, axis=0)
    reference_mean = np.asarray([[0.8, 0.2], [0.8, 0.2]], dtype=float)
    cfg = SimpleNamespace(evaluation_samples=4, probability_threshold=0.7)

    result = evaluation.evaluate_external_baselines(samples, reference_mean, cfg)[
        "bethell"
    ]

    assert result.mean_samples == pytest.approx(3.5)
    assert result.quality_rate == pytest.approx(0.5)
    assert result.metrics == {
        "class_selection": pytest.approx(1.0),
        "distribution_accuracy": pytest.approx(0.1),
        "threshold": pytest.approx(1.0),
    }


def test_mean_matched_budget_rounds_half_up_and_stays_in_stream() -> None:
    assert evaluation.mean_matched_budget(2.5, 10) == 3
    assert evaluation.mean_matched_budget(0.2, 10) == 1
    assert evaluation.mean_matched_budget(15.0, 10) == 10
