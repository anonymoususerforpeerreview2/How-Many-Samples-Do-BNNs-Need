"""Single source of configuration for CIFAR-100 runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import adaptive_cs as acs


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "datasets"
# Roughly geometric checkpoints plus the start of the final 30-draw
# early-abstention window for the common 500-draw horizon.
SPARSE_INSPECTIONS = (16, 20, 25, 32, 40, 50, 63, 80, 100, 126, 159, 200, 251, 316, 397, 470, 500)


@dataclass(frozen=True)
class Cifar100Config:
    """Paper protocol; paths come from environment, scientific choices do not."""

    run_dir: Path
    data_root: Path
    source_cache: Path | None = None
    device: str = "cuda"
    seed: int = 2026
    fit_batch_size: int = 16
    fit_workers: int = 8
    fit_max_images: int = 0
    num_inputs: int = 1_000
    calibration_size: int = 1_000
    calibration_samples: int = 50
    inference_batch_size: int = 256
    reference_samples: int = 1_000
    max_samples: int = 500
    distribution_max_samples: int = 500
    fixed_budgets: tuple[int, ...] = (5, 10, 20, 30, 50, 100, 200, 300, 400, 500)
    prior_candidates: tuple[float, ...] = (
        1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5,
        1e6, 3e6, 1e7, 3e7, 1e8,
    )
    distribution_epsilon: float = 0.05
    probability_threshold: float = 0.7
    runtime_repeats: int = 3
    cs_repeats: int = 3
    datasets: tuple[str, ...] = ("clean", "cifar100_c", "cifar100_p")
    sparse_inspections: tuple[int, ...] = SPARSE_INSPECTIONS
    smoke: bool = False

    @classmethod
    def from_environment(cls) -> "Cifar100Config":
        root = Path(__file__).resolve().parents[2]
        smoke = os.environ.get("CIFAR100_SMOKE", "0") == "1"
        run_dir = Path(
            os.environ.get("CIFAR100_RUN_DIR", root / "data" / "cifar100" / "local" / "full")
        ).expanduser()
        data_root = Path(
            os.environ.get("CIFAR100_DATA_ROOT", str(DEFAULT_DATA_ROOT))
        ).expanduser()
        source = os.environ.get("CIFAR100_SOURCE_CACHE")
        if smoke:
            return cls(
                run_dir=run_dir,
                data_root=data_root,
                source_cache=Path(source) if source else None,
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
        return cls(
            run_dir=run_dir,
            data_root=data_root,
            source_cache=Path(source) if source else None,
        )

    @property
    def fit_dir(self) -> Path:
        return self.run_dir / "fit"

    @property
    def posterior_cache(self) -> Path:
        return self.fit_dir / "posterior.pt"

    @property
    def efficiency_dir(self) -> Path:
        return self.run_dir / "sample_efficiency"

    @property
    def latency_dir(self) -> Path:
        return self.run_dir / "latency"

    @property
    def sample_dir(self) -> Path:
        return self.run_dir / "server_samples"

    def confidence_settings(self) -> acs.ConfidenceSequenceSettings:
        return acs.ConfidenceSequenceSettings(
            alpha=0.05,
            c=0.75,
            theta=0.5,
            x_tolerance=1e-4,
            y_tolerance=1e-6,
            max_steps=80,
            simplex_intersection=True,
            endpoint_root_mode="componentwise",
            root_finder="boost",
        )

    def task_max_samples(self, task: str) -> int:
        """Return the common CIFAR-100 cap for a stopping task."""
        if task == "distribution_accuracy":
            return self.distribution_max_samples
        if task in {"class_selection", "threshold"}:
            return self.max_samples
        raise ValueError(f"unknown task: {task}")

    def inspection_times_for(self, task: str) -> tuple[int, ...]:
        """Return the mandatory sparse inspection schedule for ``task``.

        Posterior draws always update the betting state, but confidence boxes
        and stopping decisions are constructed only at these checkpoints.  If
        a caller uses a shorter cap (as in a smoke test), retain the preceding
        sparse checkpoints and include that cap as the final inspection.
        """
        cap = self.task_max_samples(task)
        schedule_path = Path(self.run_dir) / "inspection_schedule.json"
        if not self.smoke and schedule_path.is_file():
            saved = json.loads(schedule_path.read_text())
            times = tuple(int(time) for time in saved["tasks"][task]["inspection_times"])
            if times != tuple(sorted(set(times))) or any(time < 1 or time > cap for time in times):
                raise ValueError(f"invalid calibrated inspection schedule for {task}")
            if times[-1] != cap:
                raise ValueError(f"calibrated inspection schedule for {task} omits the cap")
            return times
        if not self.smoke and os.environ.get("VISION_REQUIRE_CALIBRATED_SCHEDULE") == "1":
            raise FileNotFoundError(schedule_path)
        times = tuple(time for time in self.sparse_inspections if time <= cap)
        if not times or times[-1] != cap:
            times += (cap,)
        return times

    @property
    def evaluation_samples(self) -> int:
        """Number of posterior draws needed to evaluate every stopping task."""
        return max(self.max_samples, self.distribution_max_samples)

    def as_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["early_abstention_window"] = acs.EARLY_ABSTENTION_WINDOW
        result["early_abstention_late_offsets"] = list(acs.EARLY_ABSTENTION_LATE_OFFSETS)
        for key, value in list(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
        return result
