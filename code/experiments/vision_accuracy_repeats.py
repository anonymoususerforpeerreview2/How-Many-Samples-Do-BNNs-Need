"""Thirty paired accuracy streams and fixed-budget curves on paper inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dataclasses import replace

from experiments.adaptive_baselines import run_external_baselines
from experiments.cifar100.evaluation import run_tracker
from experiments.vision_accuracy_curves import fixed_accuracy_curve


def run(benchmark: str, dataset: str, paper_run: Path, repeats: int) -> None:
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
    prediction_cfg = (replace(cfg, inference_batch_size=24)
                      if benchmark == "imagenet" else cfg)
    original_dir = paper_run / "full" / "sample_efficiency"
    original_summary = json.loads((original_dir / "summary.json").read_text())
    original_config = original_summary["config"]
    if original_config["seed"] != cfg.seed or original_config["num_inputs"] != cfg.num_inputs:
        raise ValueError("repeat configuration differs from paper run")
    budget = original_summary["datasets"][dataset]["tasks"]["class_selection"]["mean_matched_fixed_budget"]
    with np.load(original_dir / f"{dataset}.npz") as original:
        source_indices = original["source_indices"].copy()
        labels = original["labels"].copy()

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
        raise ValueError("labels differ from the original run")

    output = paper_run / "accuracy_repeats" / f"{dataset}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ("clean", f"{benchmark}_c", f"{benchmark}_p")
    seed_base = cfg.seed + 1000 * names.index(dataset)
    adaptive_predictions = []
    fixed_predictions = []
    adaptive_draws = []
    adaptive_statuses = []
    curves = []
    baseline_accuracy = {key: [] for key in ("bethell", "uncertrans")}
    baseline_draws = {key: [] for key in baseline_accuracy}
    curve_output = paper_run / "accuracy_curves" / f"{dataset}.npz"
    curve_output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(repeats):
        seed = seed_base + 20 + 100_000 * repeat
        print(f"{benchmark} {dataset}: repeat {repeat}/{repeats}, seed {seed}", flush=True)
        if benchmark == "imagenet":
            samples_t, _, _ = predictive_stream(
                cfg if repeat == 0 else prediction_cfg, model, data.images,
                cfg.max_samples, seed, keep_samples=True
            )
        else:
            samples_t, _, _ = predictive_stream(
                model, posterior, data.images, cfg.max_samples,
                cfg.inference_batch_size, seed, keep_samples=True,
            )
        assert samples_t is not None
        samples = samples_t.numpy()
        curve = fixed_accuracy_curve(samples, labels)
        curves.append(curve)
        if repeat == 0:
            with np.load(original_dir / f"{dataset}.npz") as first:
                expected = first["class_selection_mean_matched_estimates"].argmax(axis=1)
                if not np.array_equal(samples[:budget].mean(axis=0, dtype=np.float64).argmax(axis=1), expected):
                    raise ValueError("original fixed-budget predictions differ")
                for key in baseline_accuracy:
                    baseline_accuracy[key].append(
                        100 * np.mean(first[f"baseline_{key}_estimates"].argmax(axis=1) == labels)
                    )
                    baseline_draws[key].append(float(first[f"baseline_{key}_stop_times"].mean()))
        else:
            for key, baseline in run_external_baselines(samples, max_samples=cfg.evaluation_samples).items():
                baseline_accuracy[key].append(
                    100 * np.mean(baseline.estimates.argmax(axis=1) == labels)
                )
                baseline_draws[key].append(baseline.mean_samples)
        np.savez_compressed(
            curve_output, source_indices=source_indices, labels=labels,
            seeds=np.asarray([seed_base + 20 + 100_000 * r for r in range(repeat + 1)]),
            fixed_accuracy_pct_by_run=np.asarray(curves),
            **{f"{key}_accuracy_pct_by_run": np.asarray(baseline_accuracy[key])
               for key in baseline_accuracy},
            **{f"{key}_mean_draws_by_run": np.asarray(baseline_draws[key])
               for key in baseline_draws},
        )
        if repeat == 0:
            continue
        result = run_tracker("class_selection", samples, cfg)
        adaptive_predictions.append(result.mean.numpy().argmax(axis=1))
        fixed_predictions.append(samples[:budget].mean(axis=0, dtype=np.float64).argmax(axis=1))
        adaptive_draws.append(result.n_samples.numpy())
        adaptive_statuses.append(np.asarray(result.status))
        np.savez_compressed(
            output,
            source_indices=source_indices,
            labels=labels,
            fixed_budget=budget,
            seeds=np.asarray([seed_base + 20 + 100_000 * r for r in range(1, repeat + 1)]),
            adaptive_predictions=np.asarray(adaptive_predictions),
            fixed_predictions=np.asarray(fixed_predictions),
            adaptive_draws=np.asarray(adaptive_draws),
            adaptive_statuses=np.asarray(adaptive_statuses),
        )
        print(f"Saved {output} through repeat {repeat}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("cifar10", "cifar100", "imagenet"))
    parser.add_argument("dataset")
    parser.add_argument("paper_run", type=Path)
    parser.add_argument("--repeats", type=int, default=30,
                        help="total streams including the original")
    args = parser.parse_args()
    run(args.benchmark, args.dataset, args.paper_run, args.repeats)


if __name__ == "__main__":
    main()
