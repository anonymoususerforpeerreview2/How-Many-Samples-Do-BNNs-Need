"""Configuration for clean/C/P CIFAR-10 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from experiments.cifar100.config import Cifar100Config


@dataclass(frozen=True)
class Cifar10Config(Cifar100Config):
    """CIFAR-10 protocol matching the paper's CIFAR-100 experiment."""

    datasets: tuple[str, ...] = ("clean", "cifar10_c", "cifar10_p")

    @classmethod
    def from_environment(cls) -> "Cifar10Config":
        root = Path(__file__).resolve().parents[2]
        smoke = os.environ.get("CIFAR10_SMOKE", "0") == "1"
        run_dir = Path(
            os.environ.get("CIFAR10_RUN_DIR", root / "data" / "cifar10" / "local" / "full")
        ).expanduser()
        data_root = Path(
            os.environ.get("CIFAR10_DATA_ROOT", str(root / "data" / "datasets"))
        ).expanduser()
        source = os.environ.get("CIFAR10_SOURCE_CACHE")
        common = {
            "run_dir": run_dir,
            "data_root": data_root,
            "source_cache": Path(source).expanduser() if source else None,
        }
        if not smoke:
            return cls(**common)
        return cls(
            **common,
            fit_batch_size=8,
            fit_workers=2,
            fit_max_images=128,
            num_inputs=24,
            calibration_size=24,
            calibration_samples=2,
            inference_batch_size=12,
            reference_samples=2,
            max_samples=4,
            distribution_max_samples=4,
            fixed_budgets=(2, 4),
            prior_candidates=(1e5, 3e5),
            runtime_repeats=1,
            cs_repeats=1,
            sparse_inspections=(2, 4),
            smoke=True,
        )
