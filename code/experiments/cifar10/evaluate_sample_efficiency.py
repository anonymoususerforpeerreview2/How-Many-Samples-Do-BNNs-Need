"""Evaluate all stopping rules on clean CIFAR-10, CIFAR-10-C, and CIFAR-10-P."""

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
from .config import Cifar10Config
from .data import evaluation_dataset
from .posterior import load_laplace_cache, load_model, predictive_stream


def main() -> None:
    cfg = Cifar10Config.from_environment()
    cfg.efficiency_dir.mkdir(parents=True, exist_ok=True)
    cfg.sample_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not cfg.posterior_cache.exists():
        raise FileNotFoundError(f"missing {cfg.posterior_cache}; run fit_laplace first")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = load_model(device)
    approximation, cache = load_laplace_cache(model, cfg.posterior_cache, device)
    summary: dict[str, object] = {
        "stage": "sample_efficiency",
        "benchmark": "cifar10",
        "model": "edadaltocg/resnet18_cifar10",
        "config": cfg.as_json(),
        "prior_precision": cache["prior_calibration"]["selected"]["prior_precision"],
        "datasets": {},
    }
    for dataset_index, dataset_name in enumerate(cfg.datasets):
        print(f"Evaluating {dataset_name}...", flush=True)
        data = evaluation_dataset(dataset_name, cfg.data_root, cfg.num_inputs, cfg.seed, device)
        seed_base = cfg.seed + 1_000 * dataset_index
        _, reference_mean_t, reference_seconds = predictive_stream(
            model, approximation, data.images, cfg.reference_samples,
            cfg.inference_batch_size, seed_base + 10, keep_samples=False,
        )
        samples_t, _, evaluation_seconds = predictive_stream(
            model, approximation, data.images, cfg.evaluation_samples,
            cfg.inference_batch_size, seed_base + 20, keep_samples=True,
        )
        assert samples_t is not None
        samples = samples_t.numpy()
        reference_mean = reference_mean_t.numpy()
        labels = data.labels.detach().cpu().numpy()
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
    save_json(cfg.efficiency_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
