"""Generate paired fixed/adaptive efficiency plots for all three tasks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRAPH_DIR = Path(__file__).resolve().parent
ROOT = GRAPH_DIR.parents[1]
sys.path.insert(0, str(ROOT / "code"))

from bnn_demo_lib.plotting import COLORS, PAPER_LINE_WIDTH, configure_matplotlib  # noqa: E402
from experiments.toy import load_config, make_toy_data, run_baselines, run_task, write_tex_macros  # noqa: E402


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def main() -> None:
    configure_matplotlib()
    cfg = load_config()
    data = make_toy_data(cfg)
    results = {
        task: run_task(task, data, cfg)
        for task in ("class_selection", "distribution_accuracy", "threshold")
    }
    baselines = {task: run_baselines(task, data, cfg) for task in results}

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.65))
    panels = (
        ("class_selection", r"$\Pr(\hat{c}_t = c^\star)$", r"(a) Class selection ($\uparrow$)"),
        ("distribution_accuracy", r"$\|\widehat \boldsymbol{\mu}_t -\boldsymbol{\mu}\|_\infty$", r"(b) Distribution accuracy ($\downarrow$)"),
        ("threshold", r"$\Pr(\mathrm{correct\ threshold\ decision})$", r"(c) Threshold decision ($\uparrow$)"),
    )
    for axis, (task, ylabel, title) in zip(axes, panels):
        result = results[task]
        axis.plot(
            result.fixed_budgets,
            result.fixed_metric,
            marker="o",
            color="black",
            label="Fixed budget",
        )
        axis.scatter(
            result.adaptive_mean_samples,
            result.adaptive_all_input_metric,
            marker="x",
            s=95,
            linewidths=PAPER_LINE_WIDTH,
            color=COLORS[1],
            alpha=0.45,
            label="Adaptive CS (all inputs)",
            zorder=3,
        )
        axis.scatter(
            result.adaptive_mean_samples,
            result.adaptive_metric,
            marker="x",
            s=95,
            linewidths=PAPER_LINE_WIDTH,
            color=COLORS[1],
            label="Adaptive CS (only resolved)",
            zorder=4,
        )
        for key, marker, color in (
            ("bethell", "s", COLORS[2]),
            ("uncertrans", "D", "#5B3F99"),
        ):
            baseline = baselines[task][key]
            axis.scatter(
                baseline.mean_samples,
                baseline.metric,
                marker=marker,
                s=58,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                label="UncerTrans" if key == "uncertrans" else baseline.name,
                zorder=4,
            )
        axis.set(xlabel="Posterior samples per input", ylabel=ylabel, title=title)
        axis.set_box_aspect(0.86)
        if task != "distribution_accuracy":
            axis.set_ylim(0.88, 1.005)
        axis.text(
            0.97 if task == "distribution_accuracy" else 0.5,
            0.95 if task == "distribution_accuracy" else 0.06,
            f"Adaptive CS:\n{_pct(result.adaptive_certified_rate)}% inputs resolved",
            transform=axis.transAxes,
            ha="right" if task == "distribution_accuracy" else "center",
            va="top" if task == "distribution_accuracy" else "bottom",
            multialignment="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7"},
        )
        axis.set_xticks((100, 200, 300, 400))
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(
        handles, labels, frameon=False, fontsize=10,
        loc="center left", bbox_to_anchor=(1.04, 0.5),
    )
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.2, top=0.87, wspace=0.52)
    fig.savefig(
        GRAPH_DIR / "toy_sample_efficiency.pdf",
        dpi=200,
        transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    summary = {
        "seed": int(cfg.experiment.seed),
        "num_inputs": len(data.means),
        "max_samples": int(cfg.experiment.max_samples),
        "tasks": {
            task: {
                "fixed_budgets": result.fixed_budgets.tolist(),
                "fixed_metric": result.fixed_metric.tolist(),
                "adaptive_mean_samples": result.adaptive_mean_samples,
                "adaptive_metric": result.adaptive_metric,
                "adaptive_all_input_metric": result.adaptive_all_input_metric,
                "adaptive_unresolved_metric": result.adaptive_unresolved_metric,
                "adaptive_certified_rate": result.adaptive_certified_rate,
                "adaptive_abstained_rate": result.adaptive_abstained_rate,
                "adaptive_timeout_rate": result.adaptive_timeout_rate,
                "baselines": {
                    key: {
                        "name": baseline.name,
                        "mean_samples": baseline.mean_samples,
                        "metric": baseline.metric,
                        "quality_rate": baseline.quality_rate,
                    }
                    for key, baseline in baselines[task].items()
                },
            }
            for task, result in results.items()
        },
    }
    (GRAPH_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    class_result = results["class_selection"]
    macros = {
        "ToyClassAgreementPct": _pct(class_result.adaptive_metric),
        "ToyClassMeanSamples": f"{class_result.adaptive_mean_samples:.1f}",
        "ToyClassCertifiedPct": _pct(class_result.adaptive_certified_rate),
        "ToyClassUnresolvedPct": _pct(1.0 - class_result.adaptive_certified_rate),
        "ToyClassAbstainedPct": _pct(class_result.adaptive_abstained_rate),
        "ToyClassTimeoutPct": _pct(class_result.adaptive_timeout_rate),
        "ToyDistributionError": f"{results['distribution_accuracy'].adaptive_metric:.3f}",
        "ToyDistributionMeanSamples": f"{results['distribution_accuracy'].adaptive_mean_samples:.1f}",
        "ToyDistributionCertifiedPct": _pct(results["distribution_accuracy"].adaptive_certified_rate),
        "ToyThresholdAgreementPct": _pct(results["threshold"].adaptive_metric),
        "ToyThresholdMeanSamples": f"{results['threshold'].adaptive_mean_samples:.1f}",
        "ToyThresholdCertifiedPct": _pct(results["threshold"].adaptive_certified_rate),
    }
    write_tex_macros(GRAPH_DIR / "metrics.tex", macros)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
