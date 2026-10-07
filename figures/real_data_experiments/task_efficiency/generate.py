"""Generate CIFAR-100 distribution-accuracy and threshold-rule panels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
GRAPH_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(GRAPH_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from bnn_demo_lib.plotting import COLORS, configure_matplotlib  # noqa: E402
from common import LABELS, load_arrays, load_run, require_calibrated_results  # noqa: E402


DATASETS = ("clean", "cifar100_c", "cifar100_p")
BASELINE_STYLES = (("bethell", "s"), ("uncertrans", "D"))


def main() -> None:
    configure_matplotlib()
    run, summary, _ = load_run()
    require_calibrated_results(run, summary)
    arrays = {name: load_arrays(run, name) for name in DATASETS}
    colors = ("black", COLORS[1], COLORS[2])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.45))
    panels = (
        ("distribution_accuracy", r"Mean $\|\widehat\mu-\mu^{\rm ref}\|_\infty$", "(a) Distribution accuracy"),
        ("threshold", "Decision agreement", "(b) Threshold decision"),
    )
    for axis, (task, ylabel, title) in zip(axes, panels):
        panel_values: list[float] = []
        certification_rates: list[float] = []
        for dataset, color in zip(DATASETS, colors):
            curve = arrays[dataset][f"{task}_fixed_metric"]
            panel_values.extend(curve.tolist())
            budgets = np.arange(1, len(curve) + 1)
            result = summary["datasets"][dataset]["tasks"][task]
            certified = arrays[dataset][f"{task}_statuses"] == "stopped"
            adaptive_metric = float("nan")
            if task == "distribution_accuracy":
                errors = np.max(
                    np.abs(
                        arrays[dataset]["distribution_accuracy_estimates"]
                        - arrays[dataset]["reference_mean"]
                    ),
                    axis=1,
                )
                if certified.any():
                    adaptive_metric = float(errors[certified].mean())
            elif task == "threshold":
                threshold = summary["config"]["probability_threshold"]
                target = arrays[dataset]["reference_mean"].max(axis=1) >= threshold
                decision = arrays[dataset]["threshold_decisions"].astype(bool)
                if certified.any():
                    adaptive_metric = float(np.mean((decision == target)[certified]))
            if np.isfinite(adaptive_metric):
                panel_values.append(adaptive_metric)
            axis.plot(budgets, curve, color=color, label=LABELS[dataset])
            axis.scatter(
                result["adaptive_mean_samples"], adaptive_metric,
                marker="X", s=60, color=color, zorder=4,
            )
            for baseline_key, marker in BASELINE_STYLES:
                baseline = result.get("baselines", {}).get(baseline_key)
                if baseline is None:
                    continue
                baseline_metric = baseline["metrics"][task]
                panel_values.append(baseline_metric)
                axis.scatter(
                    baseline["mean_samples"], baseline_metric,
                    marker=marker, s=48, color=color, edgecolors="white",
                    linewidths=0.6, zorder=5,
                )
            certification_rates.append(100 * result["adaptive_certified_rate"])
        task_cap = (
            summary["config"].get(
                "distribution_max_samples", summary["config"]["max_samples"]
            )
            if task == "distribution_accuracy"
            else summary["config"]["max_samples"]
        )
        axis.set(
            xlabel="Posterior samples per input",
            ylabel=ylabel,
            title=title,
            xlim=(1, task_cap),
        )
        if task == "threshold":
            axis.set_ylim(max(0.0, min(panel_values) - 0.03), 1.005)
        text_y = (0.82, 0.74, 0.66) if task == "distribution_accuracy" else (0.30, 0.22, 0.14)
        axis.text(
            0.98,
            text_y[0] + 0.08,
            "Certified:",
            transform=axis.transAxes,
            ha="right",
            fontsize=6.5,
            color="0.25",
        )
        for y, dataset, color, rate in zip(
            text_y, DATASETS, colors, certification_rates
        ):
            axis.text(
                0.98,
                y,
                f"{LABELS[dataset]}: {rate:.0f}%",
                transform=axis.transAxes,
                ha="right",
                fontsize=6.5,
                color=color,
            )
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend(
        (
            Line2D([], [], color="0.25", marker="X", linestyle="None"),
            Line2D([], [], color="0.25", marker="s", linestyle="None"),
            Line2D([], [], color="0.25", marker="D", linestyle="None"),
        )
    )
    labels.extend(("Adaptive CS", "Bethell et al.", "UncerTrans entropy"))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        title="Color: dataset; line: fixed; marker: adaptive method",
        title_fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.savefig(HERE / "cifar100_task_efficiency.pdf", dpi=200, transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    # A distribution-level view of the same matched-budget comparison.  Each
    # panel uses a separate dataset but the same paired, CS-certified inputs
    # for the fixed and dynamic methods.
    density_fig, density_axes = plt.subplots(1, len(DATASETS), figsize=(11.2, 3.1))
    for axis, dataset, color in zip(density_axes, DATASETS, colors):
        status = arrays[dataset]["distribution_accuracy_statuses"]
        certified = status == "stopped"
        reference = arrays[dataset]["reference_mean"]
        adaptive_errors = np.max(
            np.abs(
                arrays[dataset]["distribution_accuracy_estimates"][certified]
                - reference[certified]
            ),
            axis=1,
        )
        fixed_errors = np.max(
            np.abs(
                arrays[dataset]["distribution_accuracy_mean_matched_estimates"][certified]
                - reference[certified]
            ),
            axis=1,
        )
        bins = np.histogram_bin_edges(
            np.concatenate((fixed_errors, adaptive_errors)), bins="fd"
        )
        if len(bins) < 2:
            bins = 12
        budget = summary["datasets"][dataset]["tasks"]["distribution_accuracy"][
            "mean_matched_fixed_budget"
        ]
        axis.hist(
            fixed_errors, bins=bins, density=True, histtype="stepfilled",
            alpha=0.42, color="0.20", label=rf"Fixed $T=\bar T={budget}$",
        )
        axis.hist(
            adaptive_errors, bins=bins, density=True, histtype="stepfilled",
            alpha=0.52, color=color, label="Adaptive CS (dynamic $T$)",
        )
        axis.set(
            title=LABELS[dataset],
            xlabel=r"$\|\widehat\mu_t-\mu^{\rm ref}\|_\infty$",
        )
        if axis is density_axes[0]:
            axis.set_ylabel("Density")
            axis.legend(frameon=False, fontsize=7.5)
    density_fig.tight_layout()
    density_fig.savefig(
        HERE / "cifar100_distribution_error.pdf", dpi=200, transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(density_fig)
    derived = {}
    for dataset in DATASETS:
        threshold = summary["config"]["probability_threshold"]
        threshold_target = arrays[dataset]["reference_mean"].max(axis=1) >= threshold
        threshold_decision = arrays[dataset]["threshold_decisions"].astype(bool)
        class_target = arrays[dataset]["reference_mean"].argmax(axis=1)
        class_estimate = arrays[dataset]["class_selection_estimates"].argmax(axis=1)
        class_certified = arrays[dataset]["class_selection_statuses"] == "stopped"
        distribution_certified = (
            arrays[dataset]["distribution_accuracy_statuses"] == "stopped"
        )
        threshold_certified = arrays[dataset]["threshold_statuses"] == "stopped"
        distribution_errors = np.max(
            np.abs(
                arrays[dataset]["distribution_accuracy_estimates"]
                - arrays[dataset]["reference_mean"]
            ),
            axis=1,
        )
        derived[dataset] = {
            "class_adaptive_agreement_certified": float(
                np.mean((class_estimate == class_target)[class_certified])
            ),
            "distribution_adaptive_error_certified": float(
                distribution_errors[distribution_certified].mean()
            ),
            "threshold_adaptive_agreement_certified": float(
                np.mean(
                    (threshold_decision == threshold_target)[threshold_certified]
                )
            ),
        }
    paper_datasets = json.loads(json.dumps(summary["datasets"]))
    derived_keys = {
        "class_selection": "class_adaptive_agreement_certified",
        "distribution_accuracy": "distribution_adaptive_error_certified",
        "threshold": "threshold_adaptive_agreement_certified",
    }
    for dataset in DATASETS:
        for task, key in derived_keys.items():
            metric = derived[dataset][key]
            paper_datasets[dataset]["tasks"][task]["adaptive_metric"] = metric
            paper_datasets[dataset]["tasks"][task]["certified_metric"] = metric
    (HERE / "summary.json").write_text(
        json.dumps(
            {
                "source_run": str(run),
                "metric_policy": (
                    "adaptive quality metrics use certified inputs only; "
                    "allocation uses all inputs"
                ),
                "datasets": paper_datasets,
                "derived": derived,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Generated CIFAR-100 task figure from {run}")


if __name__ == "__main__":
    main()
