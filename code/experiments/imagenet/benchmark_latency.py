"""Benchmark all clean ImageNet inputs for each adaptive stopping task."""

from __future__ import annotations

from contextlib import nullcontext
import json

import numpy as np
import torch

from experiments.cifar100.io import save_json
from experiments.vision_latency import benchmark_fixed_budgets, benchmark_tasks
from .config import ImageNetConfig
from .data import evaluation_dataset
from .model import configure_mc_dropout, load_model, sync


@torch.inference_mode()
def main() -> None:
    cfg = ImageNetConfig.from_environment()
    cfg.latency_dir.mkdir(parents=True, exist_ok=True)
    samples = np.load(cfg.sample_dir / "clean.npy", mmap_mode="r")
    device = torch.device(cfg.device)
    model = load_model(cfg, device)
    probability = getattr(cfg, "dropout_probability", None)
    if probability is None:
        dropout = json.loads(cfg.dropout_summary.read_text(encoding="utf-8"))
        probability = float(dropout["selected"]["dropout_probability"])
    configure_mc_dropout(model, probability)
    images = evaluation_dataset("clean", cfg, cfg.num_inputs, cfg.seed, device).images
    if len(images) != samples.shape[1]:
        raise ValueError("clean images and saved prediction streams disagree")
    with (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.mixed_precision and device.type == "cuda"
        else nullcontext()
    ):
        model(images[:1])
    sync(device)

    def inference_action(schedule: np.ndarray) -> None:
        torch.manual_seed(cfg.seed + 20)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed + 20)
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if cfg.mixed_precision and device.type == "cuda"
            else nullcontext()
        ):
            for index, budget in enumerate(schedule):
                image = images[index : index + 1]
                for _ in range(int(budget)):
                    model(image).softmax(dim=1)

    fixed = benchmark_fixed_budgets(cfg, len(images), inference_action, lambda: sync(device))
    if fixed is not None:
        save_json(cfg.run_dir / "fixed_budget_latency" / "summary.json", {
            "stage": "fixed_budget_latency", "benchmark": "imagenet",
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
        "benchmark": "imagenet",
        "model": cfg.model_id,
        "dropout_probability": cfg.dropout_probability,
        "config": cfg.as_json(),
        "gpu": torch.cuda.get_device_name(device),
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
