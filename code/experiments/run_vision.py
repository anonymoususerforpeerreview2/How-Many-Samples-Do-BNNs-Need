"""Reproduce the local CIFAR-10, CIFAR-100, and ImageNet-1K runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.vision_fixed_budgets import TASKS, budgets_from_base


CODE = Path(__file__).resolve().parents[1]
BENCHMARKS = ("cifar10", "cifar100", "imagenet")


def _stage(module: str, *args: str, env: dict[str, str]) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=CODE, env=env, check=True)


def run(benchmark: str) -> None:
    run_root = CODE / "data" / benchmark / "local"
    full = run_root / "full"
    full.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[f"{benchmark.upper()}_RUN_DIR"] = str(full)
    env[f"{benchmark.upper()}_DATA_ROOT"] = str(CODE / "data" / "datasets")

    _stage(f"experiments.{benchmark}.prepare_data", env=env)
    if benchmark != "imagenet":
        _stage(f"experiments.{benchmark}.fit_laplace", env=env)

    # The first evaluation fixes the 1,000 source identifiers. The calibration
    # pool excludes those identifiers; the second evaluation uses its schedule.
    _stage(f"experiments.{benchmark}.evaluate_sample_efficiency", env=env)
    _stage("experiments.vision_calibrate_inspections", benchmark, str(run_root), env=env)
    _stage(f"experiments.{benchmark}.evaluate_sample_efficiency", env=env)
    _stage(f"experiments.{benchmark}.benchmark_latency", env=env)

    budgets, _ = budgets_from_base(full / "sample_efficiency")
    fixed_env = {**env, "FIXED_BUDGETS": ",".join(str(budgets[task]) for task in TASKS)}
    _stage(f"experiments.{benchmark}.benchmark_latency", env=fixed_env)
    _stage("experiments.vision_batch_latency", benchmark, env=env)

    for dataset in ("clean", f"{benchmark}_c", f"{benchmark}_p"):
        _stage("experiments.vision_accuracy_repeats", benchmark, dataset, str(run_root), env=env)
        _stage("experiments.vision_accuracy_curves", benchmark, dataset, str(run_root), env=env)

    latest = CODE / "data" / benchmark / "latest_run.json"
    latest.write_text(json.dumps({"run_dir": str(run_root)}, indent=2) + "\n", encoding="utf-8")
    print(f"Finished {benchmark}; wrote {latest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", nargs="?", choices=BENCHMARKS,
                        help="run one benchmark; default: all three")
    args = parser.parse_args()
    for benchmark in (args.benchmark,) if args.benchmark else BENCHMARKS:
        run(benchmark)


if __name__ == "__main__":
    main()
