from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.cifar100.config import Cifar100Config
from experiments.vision_calibrate_inspections import quantile_schedule, select_calibration


def test_calibration_excludes_evaluation_indices(tmp_path: Path) -> None:
    saved = tmp_path / "evaluation.npz"
    np.savez(saved, source_indices=np.column_stack((np.zeros(1000, dtype=int),
                                                    np.arange(1000))))
    source_indices = np.column_stack((np.zeros(1200, dtype=int), np.arange(1200)))
    data = SimpleNamespace(images=np.arange(1200), source_indices=source_indices)
    images, chosen = select_calibration(data, saved, 33)
    assert images.tolist() == list(range(1000, 1033))
    assert chosen.shape == (33, 2)


def test_quantile_schedule_preserves_late_checks() -> None:
    times = quantile_schedule(np.arange(1, 101), 500)
    assert set((470, 485, 490, 495, 498, 500)).issubset(times)
    assert times == tuple(sorted(set(times)))


def test_config_uses_calibrated_task_schedule(tmp_path: Path) -> None:
    import json

    schedule = {task: {"inspection_times": [8, 25, 470, 500]}
                for task in ("class_selection", "distribution_accuracy", "threshold")}
    (tmp_path / "inspection_schedule.json").write_text(json.dumps({"tasks": schedule}))
    cfg = Cifar100Config(run_dir=tmp_path, data_root=tmp_path)
    assert cfg.inspection_times_for("class_selection") == (8, 25, 470, 500)
