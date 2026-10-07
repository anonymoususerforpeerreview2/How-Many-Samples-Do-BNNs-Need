"""Run the paper's bounding-box ablation on all synthetic prediction streams."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from bnn_demo_lib.settings import config_from_omegaconf
from experiments.toy import load_config, make_toy_data
from experiments.toy.bbox_benchmark import (
    _run_ablation,
    _select_ablation_indices,
    _write_ablation_outputs,
)


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Run 64 inputs once")
    parser.add_argument(
        "--mode", choices=("single", "batch"), default="batch",
        help="Batch mode compares eight-core throughput; single mode matches call patterns",
    )
    parser.add_argument("--reference-workers", type=int)
    parser.add_argument("--stage", choices=("reference", "native", "merge"), default="merge")
    args = parser.parse_args()

    cfg = load_config()
    data = make_toy_data(cfg)
    seed = int(cfg.experiment.seed)
    indices = (
        _select_ablation_indices(data.is_hard, per_difficulty=32, seed=seed)
        if args.smoke
        else np.arange(len(data.means))
    )
    trajectories = [data.samples[:, int(index)] for index in indices]
    repeats = 1 if args.smoke else 5
    warmups = 1 if args.smoke else 2
    reference_workers = (
        args.reference_workers
        if args.reference_workers is not None
        else (8 if args.mode == "batch" else 1)
    )
    if hasattr(os, "sched_getaffinity"):
        allowed_cpus = sorted(os.sched_getaffinity(0))
        print(f"CPU affinity at start: {allowed_cpus}", flush=True)
        if args.stage == "reference" and len(allowed_cpus) < reference_workers:
            raise RuntimeError("Fewer allowed CPUs than Python workers")
    print(
        f"Benchmarking {len(trajectories)} inputs x {len(data.samples)} draws, "
        f"{repeats} repeat(s), {args.mode} mode, {args.stage} stage, "
        f"{reference_workers} Python worker(s) on "
        f"{platform.node()} ({_cpu_model()})",
        flush=True,
    )
    if args.stage == "native":
        print(f"Native OpenMP thread setting: {os.environ.get('OMP_NUM_THREADS')}; "
              f"PyTorch intra-op threads: {torch.get_num_threads()}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_metadata = {
        "selected_indices": indices.tolist(),
        "repeats": repeats,
        "max_samples": int(cfg.experiment.max_samples),
        "reference_workers": reference_workers,
        "mode": args.mode,
    }
    if args.stage in ("reference", "native"):
        results = _run_ablation(
            config_from_omegaconf(cfg), trajectories, repeats, warmups, seed,
            batch_inputs=args.mode == "batch",
            reference_workers=reference_workers if args.stage == "reference" else 1,
            variant_indices=(0,) if args.stage == "reference" else (1, 2, 3, 4),
        )
        stage_path = args.output_dir / f"{args.stage}.json"
        stage_path.write_text(
            json.dumps({**stage_metadata, "results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {stage_path}", flush=True)
        return
    stages = [
        json.loads((args.output_dir / f"{stage}.json").read_text(encoding="utf-8"))
        for stage in ("reference", "native")
    ]
    for stage in stages:
        for key, value in stage_metadata.items():
            if stage[key] != value:
                raise RuntimeError(f"Benchmark stage mismatch: {key}")
    results = stages[0]["results"] + stages[1]["results"]
    if [row["id"] for row in results] != [
        "reference", "cpp-bisection", "cpp-bisection-componentwise",
        "cpp-brent-componentwise", "cpp-boost-componentwise",
    ]:
        raise RuntimeError("Benchmark stages have unexpected variants")
    baseline = results[0]["median_seconds"]
    for row in results:
        row["speedup_vs_reference"] = baseline / row["median_seconds"]
    _write_ablation_outputs(
        results,
        args.output_dir / "summary.json",
        args.output_dir / "bbox_speedup.pdf",
        num_classes=int(cfg.dataset.num_classes),
        max_samples=int(cfg.experiment.max_samples),
        trajectories=len(trajectories),
        easy_trajectories=int(np.count_nonzero(~data.is_hard[indices])),
        hard_trajectories=int(np.count_nonzero(data.is_hard[indices])),
        repeats=repeats,
        warmups=warmups,
        seed=seed,
        selected_indices=indices.tolist(),
        execution_mode="single_input" if args.mode == "single" else "batched_inputs",
        reference_workers=reference_workers,
        environment={
            "host": platform.node(),
            "cpu_model": _cpu_model(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    )


if __name__ == "__main__":
    main()
