"""Generate the CIFAR-100 sample-efficiency artifacts from local results."""

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
from common import (  # noqa: E402
    LABELS,
    load_arrays,
    load_run,
    pct,
    require_calibrated_results,
    write_macros,
)


DATASETS = ("clean", "cifar100_c", "cifar100_p")
BASELINE_STYLES = (("bethell", "s"), ("uncertrans", "D"))


def _class_metric(arrays: dict[str, np.ndarray]) -> float:
    target = arrays["reference_mean"].argmax(axis=1)
    correct = arrays["class_selection_estimates"].argmax(axis=1) == target
    certified = arrays["class_selection_statuses"] == "stopped"
    return float(correct[certified].mean()) if certified.any() else float("nan")


def _class_all_input_metric(arrays: dict[str, np.ndarray]) -> float:
    target = arrays["reference_mean"].argmax(axis=1)
    estimate = arrays["class_selection_estimates"].argmax(axis=1)
    return float(np.mean(estimate == target))


def _threshold_metric(arrays: dict[str, np.ndarray], threshold: float) -> float:
    target = arrays["reference_mean"].max(axis=1) >= threshold
    decision = arrays["threshold_decisions"].astype(bool)
    certified = arrays["threshold_statuses"] == "stopped"
    return (
        float(np.mean((decision == target)[certified]))
        if certified.any()
        else float("nan")
    )


def _distribution_metric(arrays: dict[str, np.ndarray]) -> float:
    errors = np.max(
        np.abs(
            arrays["distribution_accuracy_estimates"] - arrays["reference_mean"]
        ),
        axis=1,
    )
    certified = arrays["distribution_accuracy_statuses"] == "stopped"
    return float(errors[certified].mean()) if certified.any() else float("nan")


def _scientific_tex(value: float) -> str:
    coefficient, exponent = f"{value:.0e}".split("e")
    return rf"{coefficient}\times10^{{{int(exponent)}}}"


def _prior_table(fit: dict) -> str:
    candidates = fit["prior_calibration"]["candidates"]
    selected = fit["prior_calibration"]["selected"]["prior_precision"]

    def cells(row: dict) -> list[str]:
        values = [
            f"${_scientific_tex(row['prior_precision'])}$",
            f"{row['nll']:.3f}",
            f"{pct(row['accuracy'])}\\%",
            f"{row['mean_max_probability']:.3f}",
        ]
        if row["prior_precision"] == selected:
            values = [f"\\textbf{{{value}}}" for value in values]
        return values

    rows = [
        "\\begin{longtable}{rrrr@{\\qquad}rrrr}",
        "\\caption{Prior-precision sweep on the disjoint calibration subset. Max prob. is the average largest probability of the reference BMA.}",
        "\\label{tab:prior-sweep}\\\\",
        "\\toprule",
        "$\\lambda$ & NLL & Acc. & Max prob. & $\\lambda$ & NLL & Acc. & Max prob. \\\\ ",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "$\\lambda$ & NLL & Acc. & Max prob. & $\\lambda$ & NLL & Acc. & Max prob. \\\\ ",
        "\\midrule",
        "\\endhead",
    ]
    for index in range(0, len(candidates), 2):
        row = cells(candidates[index])
        row += cells(candidates[index + 1]) if index + 1 < len(candidates) else ["", "", "", ""]
        rows.append(" & ".join(row) + " \\\\")
    rows.extend(["\\bottomrule", "\\end{longtable}"])
    return "\n".join(rows) + "\n"


def main() -> None:
    configure_matplotlib()
    run, summary, latency = load_run()
    require_calibrated_results(run, summary, latency)
    fit = json.loads((run / "full" / "fit" / "summary.json").read_text(encoding="utf-8"))
    arrays = {name: load_arrays(run, name) for name in DATASETS}
    threshold = summary["config"]["probability_threshold"]
    certified_metrics = {
        dataset: {
            "class_selection": _class_metric(arrays[dataset]),
            "distribution_accuracy": _distribution_metric(arrays[dataset]),
            "threshold": _threshold_metric(arrays[dataset], threshold),
        }
        for dataset in DATASETS
    }
    colors = ("black", COLORS[1], COLORS[2])

    fig, (agreement_axis, density_axis) = plt.subplots(
        2, 1, figsize=(5.3, 6.0), gridspec_kw={"height_ratios": [1.15, 1]}
    )
    agreement_values: list[float] = []
    for dataset, color in zip(DATASETS, colors):
        curve = arrays[dataset]["class_selection_fixed_metric"]
        agreement_values.extend(curve.tolist())
        budgets = np.arange(1, len(curve) + 1)
        result = summary["datasets"][dataset]["tasks"]["class_selection"]
        adaptive_metric = certified_metrics[dataset]["class_selection"]
        all_input_metric = _class_all_input_metric(arrays[dataset])
        agreement_values.extend((adaptive_metric, all_input_metric))
        agreement_axis.plot(budgets, curve, color=color, label=LABELS[dataset])
        agreement_axis.scatter(
            result["adaptive_mean_samples"], all_input_metric,
            color=color, marker="x", s=65, linewidths=1.6, alpha=.45, zorder=4,
        )
        agreement_axis.scatter(
            result["adaptive_mean_samples"], adaptive_metric,
            color=color, marker="x", s=65, linewidths=1.6, zorder=5,
        )
        for baseline_key, marker in BASELINE_STYLES:
            baseline = result.get("baselines", {}).get(baseline_key)
            if baseline is None:
                continue
            agreement_values.append(baseline["metrics"]["class_selection"])
            agreement_axis.scatter(
                baseline["mean_samples"],
                baseline["metrics"]["class_selection"],
                color=color,
                marker=marker,
                s=48,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )
        counts = np.bincount(
            arrays[dataset]["class_selection_stop_times"].astype(int),
            minlength=len(curve) + 1,
        )[1 : len(curve) + 1].astype(float)
        counts /= counts.sum()
        bandwidth = max(1.5, len(curve) / 100)
        radius = int(np.ceil(4 * bandwidth))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
        kernel /= kernel.sum()
        density = np.convolve(np.pad(counts, radius, mode="reflect"), kernel, mode="valid")
        density_axis.plot(
            budgets, density, linewidth=1.6, color=color, label=LABELS[dataset]
        )
    agreement_axis.set(
        ylabel="Agreement with reference BMA", xlim=(1, summary["config"]["max_samples"]),
        ylim=(max(0.0, min(agreement_values) - 0.03), 1.005),
        title="(a) Decision agreement",
    )
    agreement_axis.legend(
        handles=[
            *agreement_axis.get_legend_handles_labels()[0],
            Line2D([], [], color="0.25", marker="x", linestyle="None", markersize=7,
                   alpha=.45, label="Adaptive CS (all inputs)"),
            Line2D([], [], color="0.25", marker="x", linestyle="None", markersize=7,
                   label="Adaptive CS (certified)"),
            Line2D([], [], color="0.25", marker="s", linestyle="None", label="Bethell et al."),
            Line2D([], [], color="0.25", marker="D", linestyle="None", label="UncerTrans entropy"),
        ],
        frameon=False, fontsize=7.4, ncol=2,
        title="Color: dataset; line: fixed",
    )
    density_axis.set(
        xlabel="Posterior samples per input", ylabel="Density",
        xlim=(1, summary["config"]["max_samples"]), title="(b) Adaptive allocation",
    )
    density_axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "cifar100_sample_efficiency.pdf", dpi=200, transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    clean = summary["datasets"]["clean"]["tasks"]["class_selection"]
    clean_class_metric = certified_metrics["clean"]["class_selection"]
    clean_distribution = summary["datasets"]["clean"]["tasks"]["distribution_accuracy"]
    clean_threshold = summary["datasets"]["clean"]["tasks"]["threshold"]
    clean_distribution_error = certified_metrics["clean"]["distribution_accuracy"]
    macros = {
        "CifarClassAgreementPct": pct(clean_class_metric),
        "CifarClassMeanSamples": f"{clean['adaptive_mean_samples']:.1f}",
        "CifarClassCertifiedPct": pct(clean["adaptive_certified_rate"]),
        "CifarClassUnresolvedPct": pct(1.0 - clean["adaptive_certified_rate"]),
        "CifarClassQOneSamples": f"{clean['adaptive_q25_samples']:.0f}",
        "CifarClassMedianSamples": f"{clean['adaptive_median_samples']:.0f}",
        "CifarClassQThreeSamples": f"{clean['adaptive_q75_samples']:.0f}",
        "CifarDistributionError": f"{clean_distribution_error:.4f}",
        "CifarDistributionMeanSamples": f"{clean_distribution['adaptive_mean_samples']:.1f}",
        "CifarDistributionCertifiedPct": pct(clean_distribution["adaptive_certified_rate"]),
        "CifarThresholdAgreementPct": pct(
            certified_metrics["clean"]["threshold"]
        ),
        "CifarThresholdMeanSamples": f"{clean_threshold['adaptive_mean_samples']:.1f}",
        "CifarThresholdCertifiedPct": pct(clean_threshold["adaptive_certified_rate"]),
        "CifarPriorPrecision": _scientific_tex(summary["prior_precision"]),
        "CifarCurvatureSeconds": f"{fit['fit_timing']['curvature_seconds']:.1f}",
        "CifarCurvatureMinutes": f"{fit['fit_timing']['curvature_seconds'] / 60:.1f}",
    }
    for dataset in DATASETS[1:]:
        key = "CifarC" if dataset == "cifar100_c" else "CifarP"
        task = summary["datasets"][dataset]["tasks"]["class_selection"]
        class_metric = certified_metrics[dataset]["class_selection"]
        macros[key + "AgreementPct"] = pct(class_metric)
        macros[key + "MeanSamples"] = f"{task['adaptive_mean_samples']:.1f}"
        macros[key + "CertifiedPct"] = pct(task["adaptive_certified_rate"])
    write_macros(HERE / "metrics.tex", macros)
    (HERE / "prior_table.tex").write_text(_prior_table(fit), encoding="utf-8")
    paper_summary = json.loads(json.dumps(summary))
    for dataset, task_metrics in certified_metrics.items():
        for task, metric in task_metrics.items():
            paper_summary["datasets"][dataset]["tasks"][task]["adaptive_metric"] = metric
            paper_summary["datasets"][dataset]["tasks"][task]["certified_metric"] = metric
        paper_summary["datasets"][dataset]["tasks"]["class_selection"]["all_input_metric"] = (
            _class_all_input_metric(arrays[dataset])
        )
    (HERE / "summary.json").write_text(
        json.dumps(
            {
                "source_run": str(run),
                "metric_policy": (
                    "adaptive quality metrics use certified inputs only; "
                    "additional crosses use final estimates on all inputs; "
                    "allocation and latency use all inputs"
                ),
                "certified_adaptive_metrics": certified_metrics,
                "fit": fit,
                "sample_efficiency": paper_summary,
                "latency": latency,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Generated CIFAR-100 figure and tables from {run}")


if __name__ == "__main__":
    main()
