"""Benchmark all clean CIFAR-100 inputs for each adaptive stopping task."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters

from experiments.vision_latency import benchmark_fixed_budgets, benchmark_tasks
from .config import Cifar100Config
from .data import evaluation_dataset
from .io import save_json
from .posterior import load_laplace_cache, load_model, sync


@torch.inference_mode()
def main() -> None:
    cfg = Cifar100Config.from_environment()
    cfg.latency_dir.mkdir(parents=True, exist_ok=True)
    samples = np.load(cfg.sample_dir / "clean.npy", mmap_mode="r")
    device = torch.device(cfg.device)
    model = load_model(device)
    approximation, _ = load_laplace_cache(model, cfg.posterior_cache, device)
    images = evaluation_dataset("clean", cfg.data_root, cfg.num_inputs, cfg.seed, device).images
    if len(images) != samples.shape[1]:
        raise ValueError("clean images and saved prediction streams disagree")
    model(images[:1])
    sync(device)

    def inference_action(schedule: np.ndarray) -> None:
        generator = torch.Generator(device=device).manual_seed(cfg.seed + 20)
        try:
            for index, budget in enumerate(schedule):
                image = images[index : index + 1]
                for _ in range(int(budget)):
                    parameter_sample = approximation.sample(1, generator=generator)[0]
                    vector_to_parameters(parameter_sample, approximation.params)
                    model(image).softmax(dim=1)
        finally:
            vector_to_parameters(approximation.mean, approximation.params)

    fixed = benchmark_fixed_budgets(cfg, len(images), inference_action, lambda: sync(device))
    if fixed is not None:
        save_json(cfg.run_dir / "fixed_budget_latency" / "summary.json", {
            "stage": "fixed_budget_latency", "benchmark": "cifar100",
            "config": cfg.as_json(), "gpu": torch.cuda.get_device_name(device),
            "benchmark_type": "single-request latency", "batch_size": 1,
            "num_runtime_inputs": len(images), "tasks": fixed,
        })
        return
    tasks = benchmark_tasks(
        cfg,
        samples,
        cfg.efficiency_dir / "clean.npz",
        inference_action,
        lambda: sync(device),
    )
    summary = {
        "stage": "latency",
        "benchmark": "cifar100",
        "config": cfg.as_json(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "benchmark_type": "single-request latency",
        "batch_size": 1,
        "num_runtime_inputs": len(images),
        "selection": "all seeded clean evaluation inputs",
        "timing_method": "independent inference replay and CS timing, summed per input",
        "tasks": tasks,
    }
    save_json(cfg.latency_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
