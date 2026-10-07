"""Learn task-specific inspection times from 100 held-out vision inputs.

The pool contains 34 clean, 33 C, and 33 P inputs. Each source identifier is
checked against the 1,000 evaluation inputs before any predictions are made.
Stopping is inspected at every draw on this pool; only the resulting quantiles
are used on the evaluation inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cifar100.evaluation import TASKS, run_tracker


def _keys(indices: np.ndarray) -> set[object]:
    values = np.asarray(indices)
    if values.ndim == 2:
        return {tuple(int(item) for item in row) for row in values}
    return {str(item) for item in values} if values.dtype.kind in "US" else set(values.tolist())


def select_calibration(data, original: Path, count: int):
    with np.load(original) as saved:
        excluded = _keys(saved["source_indices"])
    values = np.asarray(data.source_indices)
    keep = [index for index in range(len(values))
            if not (_keys(values[index:index + 1]) & excluded)]
    if len(keep) < count:
        raise ValueError(f"only {len(keep)} disjoint candidates; need {count}")
    selected = np.asarray(keep[:count])
    chosen = values[selected]
    if _keys(chosen) & excluded:
        raise AssertionError("calibration and evaluation inputs overlap")
    return data.images[selected], chosen


def quantile_schedule(stop_times: np.ndarray, cap: int) -> tuple[int, ...]:
    percentiles = np.arange(5, 100, 5)
    quantiles = np.quantile(stop_times, percentiles / 100, method="higher")
    tail = (cap - 30, cap - 15, cap - 10, cap - 5, cap - 2, cap)
    return tuple(sorted({int(value) for value in quantiles} | set(tail)))


def run(benchmark: str, original_run: Path) -> None:
    if benchmark == "cifar10":
        from experiments.cifar10.config import Cifar10Config as Config
        from experiments.cifar10.data import evaluation_dataset
        from experiments.cifar10.posterior import load_laplace_cache, load_model, predictive_stream
    elif benchmark == "cifar100":
        from experiments.cifar100.config import Cifar100Config as Config
        from experiments.cifar100.data import evaluation_dataset
        from experiments.cifar100.posterior import load_laplace_cache, load_model, predictive_stream
    elif benchmark == "imagenet":
        from experiments.imagenet.config import ImageNetConfig as Config
        from experiments.imagenet.data import evaluation_dataset
        from experiments.imagenet.model import configure_mc_dropout, load_model, predictive_stream
    else:
        raise ValueError(benchmark)

    cfg = Config.from_environment()
    if cfg.smoke or cfg.max_samples != 500 or cfg.num_inputs != 1000:
        raise ValueError("calibration requires the full paper configuration")
    device = torch.device(cfg.device)
    if benchmark == "imagenet":
        model = load_model(cfg, device, data_parallel=True)
        configure_mc_dropout(model, cfg.dropout_probability)
        posterior = None
    else:
        model = load_model(device)
        posterior, _ = load_laplace_cache(model, cfg.posterior_cache, device)

    all_streams = []
    source_ids = {}
    names = ("clean", f"{benchmark}_c", f"{benchmark}_p")
    for index, (dataset, count) in enumerate(zip(names, (34, 33, 33))):
        # A larger seeded candidate draw is filtered against the saved paper
        # selection. This works for CIFAR pair indices and ImageNet member IDs.
        if benchmark == "imagenet":
            data = evaluation_dataset(dataset, cfg, 1200, cfg.seed, device)
        else:
            data = evaluation_dataset(dataset, cfg.data_root, 1200, cfg.seed, device)
        images, identifiers = select_calibration(
            data, original_run / "full" / "sample_efficiency" / f"{dataset}.npz", count
        )
        source_ids[dataset] = identifiers.tolist()
        seed = cfg.seed + 1_000 * index + 70_000
        if benchmark == "imagenet":
            samples, _, _ = predictive_stream(cfg, model, images, cfg.max_samples,
                                               seed, keep_samples=True)
        else:
            samples, _, _ = predictive_stream(model, posterior, images, cfg.max_samples,
                                               cfg.inference_batch_size, seed,
                                               keep_samples=True)
        assert samples is not None
        all_streams.append(samples.numpy())
        print(f"calibration predictions ready: {dataset}", flush=True)
    pooled = np.concatenate(all_streams, axis=1)
    full_schedule = tuple(range(1, cfg.max_samples + 1))
    tasks = {}
    for task in TASKS:
        result = run_tracker(task, pooled, cfg, inspection_times=full_schedule)
        stops = result.n_samples.numpy()
        tasks[task] = {
            "inspection_times": quantile_schedule(stops, cfg.task_max_samples(task)),
            "calibration_stop_times": stops.tolist(),
            "calibration_statuses": list(result.status),
        }
        print(f"{task}: {tasks[task]['inspection_times']}", flush=True)
    output = {
        "benchmark": benchmark, "seed": cfg.seed, "pool_size": 100,
        "pool_composition": dict(zip(names, (34, 33, 33))),
        "source_indices": source_ids,
        "selection": "seeded candidates with saved evaluation identifiers excluded",
        "quantiles_pct": list(range(5, 100, 5)),
        "late_inspections": [470, 485, 490, 495, 498, 500],
        "tasks": tasks,
    }
    path = cfg.run_dir / "inspection_schedule.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("cifar10", "cifar100", "imagenet"))
    parser.add_argument("original_run", type=Path)
    args = parser.parse_args()
    run(args.benchmark, args.original_run)


if __name__ == "__main__":
    main()
