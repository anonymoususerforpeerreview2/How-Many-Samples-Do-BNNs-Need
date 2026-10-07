"""Evaluate all stopping rules on clean ImageNet, ImageNet-C, and ImageNet-P."""

from __future__ import annotations

import json
import numpy as np
import torch

from experiments.cifar100.evaluation import (
    TASKS,
    evaluate_external_baselines,
    evaluate_task,
    task_summary,
)
from experiments.cifar100.io import save_json
from .config import ImageNetConfig
from .data import evaluation_dataset
from .model import configure_mc_dropout, load_model, predictive_stream


def main() -> None:
    cfg = ImageNetConfig.from_environment()
    cfg.efficiency_dir.mkdir(parents=True, exist_ok=True)
    cfg.sample_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    model = load_model(cfg, device, data_parallel=True)
    dropout_modules = configure_mc_dropout(model, cfg.dropout_probability)
    summary: dict[str, object] = {
        "stage": "sample_efficiency",
        "benchmark": "imagenet",
        "model": cfg.model_id,
        "posterior": {
            "type": "MC dropout",
            "dropout_probability": cfg.dropout_probability,
            "dropout_modules": dropout_modules,
        },
        "config": cfg.as_json(),
        "datasets": {},
    }
    canonical_datasets = ("clean", "imagenet_c", "imagenet_p")
    for dataset_name in cfg.datasets:
        print(f"Evaluating {dataset_name}...", flush=True)
        data = evaluation_dataset(dataset_name, cfg, cfg.num_inputs, cfg.seed, device)
        seed_base = cfg.seed + 1_000 * canonical_datasets.index(dataset_name)
        _, reference_mean_t, reference_seconds = predictive_stream(
            cfg, model, data.images, cfg.reference_samples, seed_base + 10, keep_samples=False
        )
        samples_t, _, evaluation_seconds = predictive_stream(
            cfg, model, data.images, cfg.evaluation_samples, seed_base + 20, keep_samples=True
        )
        assert samples_t is not None
        samples = samples_t.numpy()
        reference_mean = reference_mean_t.numpy()
        labels = data.labels.cpu().numpy()
        results = {task: evaluate_task(task, samples, reference_mean, cfg) for task in TASKS}
        baselines = evaluate_external_baselines(samples, reference_mean, cfg)
        arrays: dict[str, np.ndarray] = {
            "reference_mean": reference_mean,
            "labels": labels,
            "source_indices": data.source_indices,
        }
        for task, result in results.items():
            arrays[f"{task}_fixed_metric"] = result.fixed_metric
            arrays[f"{task}_stop_times"] = result.stop_times
            arrays[f"{task}_statuses"] = result.statuses
            arrays[f"{task}_estimates"] = result.estimates
            arrays[f"{task}_mean_matched_estimates"] = result.mean_matched_estimates
            if result.decisions is not None:
                arrays[f"{task}_decisions"] = result.decisions
        for key, result in baselines.items():
            arrays[f"baseline_{key}_stop_times"] = result.stop_times
            arrays[f"baseline_{key}_estimates"] = result.estimates
            arrays[f"baseline_{key}_quality_mask"] = result.quality_mask
        np.savez_compressed(cfg.efficiency_dir / f"{dataset_name}.npz", **arrays)
        if dataset_name == "clean":
            np.save(cfg.sample_dir / "clean.npy", samples)
        dataset_summary = {
            "metadata": data.metadata,
            "reference_accuracy": float(np.mean(reference_mean.argmax(axis=1) == labels)),
            "reference_mean_seconds": reference_seconds,
            "evaluation_stream_seconds": evaluation_seconds,
            "tasks": {task: task_summary(result, cfg) for task, result in results.items()},
            "baselines": {
                key: {
                    "name": value.name,
                    "mean_samples": value.mean_samples,
                    "quality_rate": value.quality_rate,
                    "metrics": value.metrics,
                }
                for key, value in baselines.items()
            },
            "fixed": {
                str(budget): {task: float(results[task].fixed_metric[budget - 1]) for task in TASKS}
                for budget in cfg.fixed_budgets
            },
        }
        summary["datasets"][dataset_name] = dataset_summary  # type: ignore[index]
        print(json.dumps({dataset_name: dataset_summary}, indent=2), flush=True)
        del data, samples_t, samples
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(cfg.efficiency_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
