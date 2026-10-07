"""Measure label accuracy at every fixed budget on the three paper streams.

This recovers the budget curves omitted from the compact repeated-accuracy
artifacts. The seed, model, input selection, and posterior setup match
``vision_accuracy_repeats`` exactly. External allocation rules are evaluated
on each same stream, including capped inputs in their diagnostic accuracy.
ImageNet reuses its accuracy experiment's 16-input, four-draw forward batches;
the separate single-request latency benchmark has its own batch-size-one path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.adaptive_baselines import run_external_baselines


def fixed_accuracy_curve(samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Percent correct on all inputs for budgets 1 through the common cap."""
    running = np.zeros(samples.shape[1:], dtype=np.float64)
    curve = np.empty(len(samples), dtype=np.float64)
    for index, draw in enumerate(samples):
        running += draw
        curve[index] = 100 * np.mean(running.argmax(axis=1) == labels)
    return curve


def run(benchmark: str, dataset: str, paper_run: Path) -> None:
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
    original_dir = paper_run / "full" / "sample_efficiency"
    original_summary = json.loads((original_dir / "summary.json").read_text())
    original_config = original_summary["config"]
    if original_config["seed"] != cfg.seed or original_config["num_inputs"] != cfg.num_inputs:
        raise ValueError("configuration differs from the paper run")
    with np.load(original_dir / f"{dataset}.npz") as original:
        source_indices = original["source_indices"].copy()
        labels = original["labels"].copy()
        first_baseline_predictions = {
            key: original[f"baseline_{key}_estimates"].argmax(axis=1)
            for key in ("bethell", "uncertrans")
        }
        first_baseline_draws = {
            key: original[f"baseline_{key}_stop_times"].mean()
            for key in ("bethell", "uncertrans")
        }

    repeats_path = paper_run / "accuracy_repeats" / f"{dataset}.npz"
    with np.load(repeats_path) as repeats:
        if not np.array_equal(repeats["source_indices"], source_indices):
            raise ValueError("repeat source indices differ")
        if not np.array_equal(repeats["labels"], labels):
            raise ValueError("repeat labels differ")
        seeds_expected = repeats["seeds"].tolist()
        fixed_budget = int(repeats["fixed_budget"])
        fixed_expected = repeats["fixed_predictions"].copy()

    device = torch.device(cfg.device)
    if benchmark == "imagenet":
        model = load_model(cfg, device, data_parallel=True)
        configure_mc_dropout(model, cfg.dropout_probability)
        data = evaluation_dataset(dataset, cfg, cfg.num_inputs, cfg.seed, device)
        posterior = None
    else:
        model = load_model(device)
        posterior, _ = load_laplace_cache(model, cfg.posterior_cache, device)
        data = evaluation_dataset(dataset, cfg.data_root, cfg.num_inputs, cfg.seed, device)
    if not np.array_equal(data.source_indices, source_indices):
        raise ValueError("selected inputs differ from the original run")
    if not np.array_equal(data.labels.cpu().numpy(), labels):
        raise ValueError("selected labels differ from the original run")

    names = ("clean", f"{benchmark}_c", f"{benchmark}_p")
    seed_base = cfg.seed + 1000 * names.index(dataset) + 20
    seeds = [seed_base + 100_000 * repeat for repeat in range(3)]
    if seeds[1:] != seeds_expected:
        raise ValueError("repeat seeds differ from the original run")

    output = paper_run / "accuracy_curves" / f"{dataset}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    curves = []
    baseline_accuracy = {key: [] for key in first_baseline_predictions}
    baseline_draws = {key: [] for key in first_baseline_predictions}
    for repeat, seed in enumerate(seeds):
        print(f"{benchmark}/{dataset}: stream {repeat + 1}/3, seed {seed}", flush=True)
        if benchmark == "imagenet":
            samples_t, _, _ = predictive_stream(
                cfg, model, data.images, cfg.max_samples, seed, keep_samples=True
            )
        else:
            samples_t, _, _ = predictive_stream(
                model, posterior, data.images, cfg.max_samples,
                cfg.inference_batch_size, seed, keep_samples=True,
            )
        assert samples_t is not None
        samples = samples_t.numpy()
        curve = fixed_accuracy_curve(samples, labels)
        if repeat == 0:
            with np.load(original_dir / f"{dataset}.npz") as original:
                expected = original["class_selection_mean_matched_estimates"].argmax(axis=1)
        else:
            expected = fixed_expected[repeat - 1]
        actual = samples[:fixed_budget].mean(axis=0, dtype=np.float64).argmax(axis=1)
        if not np.array_equal(actual, expected):
            raise ValueError(f"stream {repeat}: fixed predictions differ from saved run")
        if curve[fixed_budget - 1] != 100 * np.mean(expected == labels):
            raise ValueError("fixed curve differs at the mean-matched budget")
        curves.append(curve)

        if repeat == 0:
            for key in baseline_accuracy:
                baseline_accuracy[key].append(100 * np.mean(first_baseline_predictions[key] == labels))
                baseline_draws[key].append(float(first_baseline_draws[key]))
        else:
            for key, result in run_external_baselines(samples, max_samples=cfg.evaluation_samples).items():
                baseline_accuracy[key].append(100 * np.mean(result.estimates.argmax(axis=1) == labels))
                baseline_draws[key].append(result.mean_samples)
        np.savez_compressed(
            output,
            source_indices=source_indices,
            labels=labels,
            seeds=np.asarray(seeds),
            fixed_accuracy_pct_by_run=np.asarray(curves),
            bethell_accuracy_pct_by_run=np.asarray(baseline_accuracy["bethell"]),
            bethell_mean_draws_by_run=np.asarray(baseline_draws["bethell"]),
            uncertrans_accuracy_pct_by_run=np.asarray(baseline_accuracy["uncertrans"]),
            uncertrans_mean_draws_by_run=np.asarray(baseline_draws["uncertrans"]),
        )
        print(f"Saved {output} through stream {repeat + 1}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("cifar10", "cifar100", "imagenet"))
    parser.add_argument("dataset")
    parser.add_argument("paper_run", type=Path)
    args = parser.parse_args()
    run(args.benchmark, args.dataset, args.paper_run)


if __name__ == "__main__":
    main()
