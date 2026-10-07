"""Checks for the common CIFAR-100 sampling horizon."""

from pathlib import Path

from experiments.cifar100.config import Cifar100Config


def test_all_cifar100_tasks_use_the_common_horizon() -> None:
    cfg = Cifar100Config(run_dir=Path("run"), data_root=Path("data"))

    assert cfg.task_max_samples("class_selection") == 500
    assert cfg.task_max_samples("threshold") == 500
    assert cfg.task_max_samples("distribution_accuracy") == 500
    assert cfg.evaluation_samples == 500
    assert len(cfg.sparse_inspections) == 17
    assert cfg.sparse_inspections[-2] == 470
    assert cfg.sparse_inspections[-1] == 500
    assert cfg.as_json()["early_abstention_window"] == 30
    assert cfg.as_json()["early_abstention_late_offsets"] == [20, 10, 5]
    for task in ("class_selection", "distribution_accuracy", "threshold"):
        assert cfg.inspection_times_for(task) == cfg.sparse_inspections


def test_sparse_schedule_is_capped_for_short_experiments() -> None:
    cfg = Cifar100Config(
        run_dir=Path("run"),
        data_root=Path("data"),
        max_samples=30,
        distribution_max_samples=30,
        sparse_inspections=(16, 20, 25, 32),
    )

    assert cfg.inspection_times_for("class_selection") == (16, 20, 25, 30)
