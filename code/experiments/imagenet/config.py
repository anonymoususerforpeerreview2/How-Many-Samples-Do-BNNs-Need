"""Configuration for clean/C/P ImageNet-1K experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from experiments.cifar100.config import Cifar100Config


@dataclass(frozen=True)
class ImageNetConfig(Cifar100Config):
    datasets: tuple[str, ...] = ("clean", "imagenet_c", "imagenet_p")
    model_id: str = "google/vit-large-patch16-384"
    dropout_probability: float = 0.10
    # Four MC samples are vectorized per image batch.  Sixteen images therefore
    # give an effective ViT batch of 64; reduce this on smaller GPUs.
    inference_batch_size: int = 16
    mc_sample_chunk: int = 4
    mixed_precision: bool = True
    runtime_repeats: int = 1
    cs_repeats: int = 1

    @classmethod
    def from_environment(cls) -> "ImageNetConfig":
        root = Path(__file__).resolve().parents[2]
        smoke = os.environ.get("IMAGENET_SMOKE", "0") == "1"
        selected_datasets = tuple(
            name.strip()
            for name in os.environ.get(
                "IMAGENET_DATASETS", "clean,imagenet_c,imagenet_p"
            ).split(",")
            if name.strip()
        )
        unknown = set(selected_datasets) - {"clean", "imagenet_c", "imagenet_p"}
        if unknown:
            raise ValueError(f"unknown IMAGENET_DATASETS entries: {sorted(unknown)}")
        common = {
            "run_dir": Path(
                os.environ.get("IMAGENET_RUN_DIR", root / "data" / "imagenet" / "local" / "full")
            ).expanduser(),
            "data_root": Path(
                os.environ.get("IMAGENET_DATA_ROOT", str(root / "data" / "datasets"))
            ).expanduser(),
            "source_cache": None,
            "datasets": selected_datasets,
        }
        if not smoke:
            return cls(**common)
        return cls(
            **common,
            num_inputs=12,
            calibration_size=12,
            reference_samples=3,
            max_samples=4,
            distribution_max_samples=4,
            fixed_budgets=(2, 4),
            sparse_inspections=(2, 4),
            inference_batch_size=2,
            mc_sample_chunk=2,
            smoke=True,
        )
