"""Choose the smallest constant draw budgets matching adaptive quality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TASKS = ("class_selection", "threshold", "distribution_accuracy")


def budgets_from_base(base: Path) -> tuple[dict[str, int], dict[str, float]]:
    config = json.loads((base / "summary.json").read_text(encoding="utf-8"))["config"]
    with np.load(base / "clean.npz") as arrays:
        reference = arrays["reference_mean"]
        threshold = config["probability_threshold"]
        metrics = {}
        selected = {}
        for task in TASKS:
            estimates = arrays[f"{task}_estimates"]
            if task == "class_selection":
                metric = float(np.mean(estimates.argmax(1) == reference.argmax(1)))
            elif task == "threshold":
                metric = float(np.mean(
                    (estimates.max(1) >= threshold) == (reference.max(1) >= threshold)
                ))
            else:
                metric = float(np.mean(np.max(np.abs(estimates - reference), axis=1)))
            curve = arrays[f"{task}_fixed_metric"]
            valid = curve <= metric + 1e-12 if task == "distribution_accuracy" else curve >= metric - 1e-12
            indices = np.flatnonzero(valid)
            if not len(indices):
                raise ValueError(f"{task}: no quality-matched fixed budget")
            metrics[task] = metric
            selected[task] = int(indices[0] + 1)
    return selected, metrics
