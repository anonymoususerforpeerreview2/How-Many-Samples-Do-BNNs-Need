from pathlib import Path

from experiments.cifar10.config import Cifar10Config


def test_cifar10_matches_cifar100_sampling_protocol() -> None:
    cfg = Cifar10Config(run_dir=Path("run"), data_root=Path("data"))
    assert cfg.datasets == ("clean", "cifar10_c", "cifar10_p")
    assert cfg.evaluation_samples == 500
    assert cfg.inspection_times_for("class_selection")[-1] == 500
