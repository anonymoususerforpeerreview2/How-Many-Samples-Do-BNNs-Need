"""Load the local CIFAR-100 run used by the paper figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LATEST = ROOT / "code" / "data" / "cifar100" / "latest_run.json"
LABELS = {"clean": "CIFAR-100", "cifar100_c": "CIFAR-100-C", "cifar100_p": "CIFAR-100-P"}


def load_run() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if not LATEST.exists():
        raise FileNotFoundError(
            f"missing {LATEST}; run `python -m experiments.run_vision` first"
        )
    run = Path(json.loads(LATEST.read_text(encoding="utf-8"))["run_dir"])
    efficiency_path = run / "full" / "sample_efficiency" / "summary.json"
    latency_path = run / "full" / "latency" / "summary.json"
    return (
        run,
        json.loads(efficiency_path.read_text(encoding="utf-8")),
        json.loads(latency_path.read_text(encoding="utf-8")),
    )


def load_arrays(run: Path, dataset: str) -> dict[str, np.ndarray]:
    path = run / "full" / "sample_efficiency" / f"{dataset}.npz"
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def require_calibrated_results(
    run: Path, summary: dict[str, Any], latency: dict[str, Any] | None = None
) -> None:
    """Check the held-out, task-specific inspection schedule."""
    schedule = json.loads((run / "full" / "inspection_schedule.json").read_text(encoding="utf-8"))
    if schedule["benchmark"] != "cifar100" or schedule["pool_size"] != 100:
        raise ValueError("missing the 100-input CIFAR-100 calibration")
    if (summary["config"].get("early_abstention_window") != 30
            or summary["config"].get("early_abstention_late_offsets") != [20, 10, 5]
            or 470 not in schedule["tasks"]["class_selection"]["inspection_times"]):
        raise ValueError("reproduce the CIFAR-100 final-30-draw evaluation first")
    for dataset, values in summary["datasets"].items():
        for task, task_values in values["tasks"].items():
            expected = list(schedule["tasks"][task]["inspection_times"])
            if task_values.get("inspection_times") != expected:
                raise ValueError(
                    f"{dataset}/{task} lacks the calibrated inspection "
                    "metadata; reproduce the CIFAR-100 evaluation first"
                )
    if latency is not None:
        if (
            latency.get("num_runtime_inputs") != summary["config"]["num_inputs"]
            or latency.get("config", {}).get("early_abstention_window") != 30
            or latency.get("config", {}).get("early_abstention_late_offsets") != [20, 10, 5]
            or set(latency.get("tasks", {})) != set(summary["datasets"]["clean"]["tasks"])
        ):
            raise ValueError(
                "latency results must cover all clean inputs and all stopping tasks"
            )


def write_macros(path: Path, values: dict[str, str]) -> None:
    lines = ["% Generated from the latest local run; do not edit."]
    lines.extend(f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{100 * value:.1f}"
