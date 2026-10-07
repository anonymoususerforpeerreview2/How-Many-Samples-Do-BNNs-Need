"""Replay adaptive and quality-matched fixed inference in batches of 16."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters

from experiments.cifar100.evaluation import TASKS, run_tracker
from experiments.cifar100.io import save_json
from experiments.vision_fixed_budgets import budgets_from_base
from experiments.vision_latency import median_runtime


def run(benchmark: str, batch_size: int = 16) -> None:
    if benchmark == "cifar10":
        from experiments.cifar10.config import Cifar10Config as Config
        from experiments.cifar10.data import evaluation_dataset
        from experiments.cifar10.posterior import load_laplace_cache, load_model, sync
    elif benchmark == "cifar100":
        from experiments.cifar100.config import Cifar100Config as Config
        from experiments.cifar100.data import evaluation_dataset
        from experiments.cifar100.posterior import load_laplace_cache, load_model, sync
    elif benchmark == "imagenet":
        from experiments.imagenet.config import ImageNetConfig as Config
        from experiments.imagenet.data import evaluation_dataset
        from experiments.imagenet.model import configure_mc_dropout, load_model, sync
    else:
        raise ValueError(benchmark)
    cfg = Config.from_environment()
    device = torch.device(cfg.device)
    base = cfg.efficiency_dir
    with np.load(base / "clean.npz") as saved:
        schedules = {task: saved[f"{task}_stop_times"].astype(np.int64) for task in TASKS}
        source_indices = saved["source_indices"].copy()
    samples = np.load(cfg.sample_dir / "clean.npy", mmap_mode="r")
    budgets, _ = budgets_from_base(base)
    if benchmark == "imagenet":
        model = load_model(cfg, device)
        configure_mc_dropout(model, cfg.dropout_probability)
        data = evaluation_dataset("clean", cfg, cfg.num_inputs, cfg.seed, device)
        posterior = None
    else:
        model = load_model(device)
        posterior, _ = load_laplace_cache(model, cfg.posterior_cache, device)
        data = evaluation_dataset("clean", cfg.data_root, cfg.num_inputs, cfg.seed, device)
    if not np.array_equal(data.source_indices, source_indices):
        raise ValueError("batch timing input selection differs from evaluation")
    images = data.images
    if len(images) != cfg.num_inputs or samples.shape[1] != cfg.num_inputs:
        raise ValueError("batch timing requires all clean evaluation inputs")
    warmup_amp = (torch.autocast(device_type="cuda", dtype=torch.float16)
                  if benchmark == "imagenet" and cfg.mixed_precision else nullcontext())
    with torch.inference_mode(), warmup_amp:
        model(images[:batch_size]).softmax(dim=1)
    sync(device)

    def inference_action(schedule: np.ndarray) -> None:
        if benchmark == "imagenet":
            torch.manual_seed(cfg.seed + 20)
            torch.cuda.manual_seed_all(cfg.seed + 20)
        else:
            generator = torch.Generator(device=device).manual_seed(cfg.seed + 20)
        amp = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if benchmark == "imagenet" and cfg.mixed_precision else nullcontext())
        try:
            with torch.inference_mode(), amp:
                for start in range(0, len(images), batch_size):
                    end = min(start + batch_size, len(images))
                    batch = images[start:end]
                    needed = schedule[start:end]
                    for draw in range(int(needed.max())):
                        active = np.flatnonzero(needed > draw)
                        if benchmark != "imagenet":
                            parameter_sample = posterior.sample(1, generator=generator)[0]
                            vector_to_parameters(parameter_sample, posterior.params)
                        model(batch[active]).softmax(dim=1)
        finally:
            if posterior is not None:
                vector_to_parameters(posterior.mean, posterior.params)

    # Match Table 2: independent forward-pass and CS timers, summed per input.
    rows = {}
    for task in TASKS:
        schedule = schedules[task]
        if np.any(schedule < 1) or np.any(schedule > cfg.task_max_samples(task)):
            raise ValueError(f"invalid {task} stop times")
        inference = median_runtime(lambda: inference_action(schedule),
                                   cfg.runtime_repeats, lambda: sync(device)) / len(images)

        def cs_action() -> None:
            for start in range(0, len(images), batch_size):
                end = min(start + batch_size, len(images))
                result = run_tracker(task, np.asarray(samples[:, start:end]), cfg)
                if not np.array_equal(result.n_samples.numpy(), schedule[start:end]):
                    raise ValueError(f"batched {task} stopping times differ")

        cs = median_runtime(cs_action, cfg.cs_repeats, lambda: None) / len(images)
        fixed_schedule = np.full(len(images), budgets[task], dtype=np.int64)
        fixed = median_runtime(lambda: inference_action(fixed_schedule),
                               cfg.runtime_repeats, lambda: sync(device)) / len(images)
        rows[task] = {
            "mean_draws": float(schedule.mean()), "fixed_draws": budgets[task],
            "inference_seconds_per_input": inference,
            "cs_seconds_per_input": cs,
            "adaptive_seconds_per_input": inference + cs,
            "fixed_seconds_per_input": fixed,
            "fixed_over_adaptive_ratio": fixed / (inference + cs),
        }
        print(f"{task}: {rows[task]}", flush=True)
    output = {
        "benchmark": benchmark, "batch_size": batch_size,
        "num_inputs": len(images), "gpu": torch.cuda.get_device_name(device),
        "timing_method": "separate batched inference replay and batched CS timing",
        "tasks": rows,
    }
    destination = cfg.run_dir / "batch_latency" / "summary.json"
    save_json(destination, output)
    print(json.dumps(output, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("cifar10", "cifar100", "imagenet"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.benchmark, args.batch_size)


if __name__ == "__main__":
    main()
