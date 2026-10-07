"""Plot repeated class-selection accuracy from the paper run and fresh streams."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))

from bnn_demo_lib.plot_style import COLORS, PAPER_LINE_WIDTH, configure_matplotlib  # noqa: E402

SPECS = (
    ("cifar10", "CifarTen", ("clean", "cifar10_c", "cifar10_p")),
    ("cifar100", "Cifar", ("clean", "cifar100_c", "cifar100_p")),
    ("imagenet", "ImageNet", ("clean", "imagenet_c", "imagenet_p")),
)

BASELINE_STYLES = {
    "bethell": ("Bethell et al.", "#2878B5"),
    "uncertrans": ("UncerTrans", "#7953A5"),
}
POINT_SIZE = 2.2
INTERVAL_CAP = 3.2
NUM_STREAMS = 30
VARIATION_XTICKS = (10, 20, 50, 100, 200, 500)
TITLES = {
    "cifar10": ("CIFAR-10", "CIFAR-10-C", "CIFAR-10-P"),
    "cifar100": ("CIFAR-100", "CIFAR-100-C", "CIFAR-100-P"),
    "imagenet": ("ImageNet-1K", "ImageNet-C", "ImageNet-P"),
}


def accuracy_sd(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sample SD of run-level accuracies, measured in percentage points."""
    return np.std(values, axis=axis, ddof=1)


def imagenet_repeats_complete() -> bool:
    """Check whether every ImageNet panel has the streams needed by this plot."""
    latest = ROOT / "code" / "data" / "imagenet" / "latest_run.json"
    if not latest.exists():
        return False
    run_dir = Path(json.loads(latest.read_text())["run_dir"])
    for dataset in SPECS[2][2]:
        repeats_path = run_dir / "accuracy_repeats" / f"{dataset}.npz"
        curves_path = run_dir / "accuracy_curves" / f"{dataset}.npz"
        if not repeats_path.exists() or not curves_path.exists():
            return False
        with np.load(repeats_path) as repeats, np.load(curves_path) as curves:
            if (repeats["adaptive_predictions"].shape[0] != NUM_STREAMS - 1
                    or curves["fixed_accuracy_pct_by_run"].shape[0] != NUM_STREAMS):
                return False
    return True


def plot_accuracy(
    rows: dict[str, dict], curves: dict[str, dict], *,
    specs: tuple = SPECS, output_name: str = "accuracy_repeats",
    pending_imagenet: bool = False,
) -> None:
    """Figure-6-style accuracy panels, with variation across streams."""
    configure_matplotlib()
    total_rows = len(SPECS) if pending_imagenet else len(specs)
    fig, axes = plt.subplots(total_rows, 3, figsize=(8.4, 2.35 * total_rows + .75))
    axes = np.atleast_2d(axes)
    for row_index, (benchmark, _, datasets) in enumerate(specs):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            key = f"{benchmark}/{dataset}"
            row = rows[key]
            data = curves[key]
            fixed = data["fixed_accuracy_pct_by_run"]
            samples = np.arange(1, fixed.shape[1] + 1)
            mean = fixed.mean(axis=0)
            sd = fixed.std(axis=0, ddof=1)
            axis.fill_between(samples, mean - sd, mean + sd, color="0.45", alpha=.18,
                              linewidth=0, zorder=1)
            axis.plot(samples, mean, color="black", linewidth=1.65, zorder=2)
            axis.errorbar(row["adaptive_mean_draws"], row["adaptive_accuracy_mean_pct"],
                          yerr=row["adaptive_accuracy_sd_pp"], fmt="o", color="#C82423",
                          markersize=POINT_SIZE, markeredgewidth=0,
                          capsize=INTERVAL_CAP, capthick=1.25,
                          elinewidth=1.25, zorder=6)
            values = np.r_[mean - sd, mean + sd,
                           row["adaptive_accuracy_mean_pct"] - row["adaptive_accuracy_sd_pp"],
                           row["adaptive_accuracy_mean_pct"] + row["adaptive_accuracy_sd_pp"]]
            for baseline, (_, color) in BASELINE_STYLES.items():
                y = data[f"{baseline}_accuracy_pct_by_run"]
                x = data[f"{baseline}_mean_draws_by_run"]
                axis.errorbar(x.mean(), y.mean(), yerr=y.std(ddof=1), fmt="o",
                              color=color, markersize=POINT_SIZE,
                              markeredgewidth=0, capsize=INTERVAL_CAP,
                              capthick=1.25, elinewidth=1.25, zorder=5)
                values = np.r_[values, y.mean() - y.std(ddof=1), y.mean() + y.std(ddof=1)]
            lower, upper = float(values.min()), float(values.max())
            margin = max(0.7, .07 * (upper - lower))
            axis.set_ylim(max(0, lower - margin), min(100, upper + margin))
            axis.set_xscale("log")
            axis.set_xlim(1, fixed.shape[1])
            axis.set_xticks((1, 10, 100, 500))
            axis.set_xticklabels(("1", "10", "100", "500"))
            axis.minorticks_off()
            axis.set_title(f"({chr(97 + 3 * row_index + column_index)}) {TITLES[benchmark][column_index]}",
                           fontsize=10, pad=5)
            if column_index == 0:
                axis.set_ylabel("Accuracy (%)", fontsize=10)
            if row_index == len(specs) - 1:
                axis.set_xlabel("Posterior samples per input", fontsize=10)
            axis.tick_params(labelsize=8, width=1.2, length=3)
            axis.spines["left"].set_linewidth(1.3)
            axis.spines["bottom"].set_linewidth(1.3)
            axis.grid(False)
    if pending_imagenet:
        for axis in axes[2]:
            axis.set_axis_off()
            axis.text(.5, .5, "ImageNet results pending", ha="center", va="center",
                      transform=axis.transAxes, fontsize=9, color="0.45")
    handles = [
        Line2D([], [], color="black", linewidth=1.65, label="Fixed budget (mean $\\pm$ SD)"),
        Line2D([], [], color="#C82423", marker="o", linestyle="None", markersize=3.2,
               label="Adaptive CS"),
        *[Line2D([], [], color=color, marker="o", linestyle="None", markersize=3.2,
                 label=label) for label, color in BASELINE_STYLES.values()],
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .015), fontsize=8.5, columnspacing=1.1,
               handletextpad=.4)
    fig.subplots_adjust(left=.075, right=.985, top=.94,
                        bottom=.19 if total_rows == 2 else .10,
                        wspace=.31, hspace=.48)
    fig.savefig(HERE / f"{output_name}.pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(HERE / f"{output_name}.png", dpi=240, transparent=True)
    plt.close(fig)


def plot_cifar_c_precision(rows: dict[str, dict], curves: dict[str, dict]) -> None:
    """Show the run-to-run accuracy spread on CIFAR-10-C."""
    configure_matplotlib()
    # The left density panel is 3.4 in wide in a .42\linewidth minipage.
    # This width makes the two panels the same size after LaTeX scales them.
    fig, axis = plt.subplots(figsize=(3.4 * .55 / .42, 2.85))
    row, data = rows["cifar10/cifar10_c"], curves["cifar10/cifar10_c"]
    fixed = data["fixed_accuracy_pct_by_run"]
    samples = np.arange(10, fixed.shape[1] + 1)
    axis.plot(samples, accuracy_sd(fixed[:, 9:]), color="black",
              linewidth=1.7, zorder=2)
    methods = (
        (row["adaptive_mean_draws"], row["adaptive_accuracy_pct_by_run"],
         COLORS[1], "x", 95, .45, "Adaptive CS\n(all inputs)"),
        (row["bethell_mean_draws"], data["bethell_accuracy_pct_by_run"],
         COLORS[2], "s", 58, 1., "Bethell et al."),
        (row["uncertrans_mean_draws"], data["uncertrans_accuracy_pct_by_run"],
         "#5B3F99", "D", 58, 1., "UncerTrans"),
    )
    handles = [Line2D([], [], color="black", marker="o", linewidth=1.7,
                      markersize=6, label="Fixed budget")]
    for draws, accuracies, color, marker, size, alpha, label in methods:
        handles.append(axis.scatter(
            draws, accuracy_sd(accuracies), marker=marker, s=size,
            linewidths=PAPER_LINE_WIDTH if marker == "x" else 0,
            color=color, edgecolors="none" if marker != "x" else None,
            alpha=alpha, label=label, zorder=5,
        ))
    axis.set_xscale("log")
    axis.set_xlim(10, 500)
    axis.set_xticks(VARIATION_XTICKS)
    axis.set_xticklabels(tuple(map(str, VARIATION_XTICKS)))
    axis.minorticks_off()
    axis.set_ylim(0, .85)
    axis.set_yticks((0, .2, .4, .6, .8))
    axis.set_xlabel("Posterior samples per input (log scale)")
    axis.set_ylabel("Accuracy SD (%)")
    # r"$\Pr(\hat{c}_t = c^\star)$", r"(a) Class selection ($\uparrow$)")
    axis.set_title(r"Run-to-run accuracy variation ($\downarrow$)", fontsize=17.5, pad=6)
    axis.grid(False)
    axis.legend(handles=handles, loc="center left", ncol=1, frameon=False,
                bbox_to_anchor=(1.04, .63), fontsize=9.5, labelspacing=.5,
                handletextpad=.35, borderaxespad=0)
    fig.subplots_adjust(left=.149, right=.727, top=.864, bottom=.215)
    fig.savefig(HERE / "accuracy_precision_cifar10_c.pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(HERE / "accuracy_precision_cifar10_c.png", dpi=240,
                transparent=True)
    plt.close(fig)


def plot_accuracy_variation(
    rows: dict[str, dict], curves: dict[str, dict], *,
    pending_imagenet: bool = False,
) -> None:
    """Figure-8-style SD panels for the full 3x3 benchmark layout."""
    configure_matplotlib()
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 7.7))
    plotted_axes = []
    max_sd = 0.0
    for row_index, (benchmark, _, datasets) in enumerate(SPECS):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            title = f"({chr(97 + 3 * row_index + column_index)}) {TITLES[benchmark][column_index]}"
            if pending_imagenet and benchmark == "imagenet":
                axis.set_axis_off()
                axis.text(.5, .55, title, ha="center", va="center",
                          transform=axis.transAxes, fontsize=10)
                axis.text(.5, .43, "Results pending", ha="center", va="center",
                          transform=axis.transAxes, fontsize=9, color="0.45")
                continue
            key = f"{benchmark}/{dataset}"
            row, data = rows[key], curves[key]
            fixed = data["fixed_accuracy_pct_by_run"]
            samples = np.arange(10, fixed.shape[1] + 1)
            fixed_sd = accuracy_sd(fixed[:, 9:])
            axis.plot(samples, fixed_sd, color="black", linewidth=1.55, zorder=2)
            methods = (
                (row["adaptive_mean_draws"], row["adaptive_accuracy_sd_pp"],
                 COLORS[1], "x", 34),
                (row["bethell_mean_draws"], row["bethell_accuracy_sd_pp"],
                 COLORS[2], "s", 23),
                (row["uncertrans_mean_draws"], row["uncertrans_accuracy_sd_pp"],
                 "#5B3F99", "D", 23),
            )
            for draws, sd, color, marker, size in methods:
                axis.scatter(draws, sd, marker=marker, s=size, color=color,
                             linewidths=1.3 if marker == "x" else 0,
                             edgecolors="none" if marker != "x" else None,
                             zorder=5)
            max_sd = max(max_sd, float(fixed_sd.max()),
                         *(sd for _, sd, _, _, _ in methods))
            plotted_axes.append(axis)
            axis.set_xscale("log")
            axis.set_xlim(10, fixed.shape[1])
            axis.set_xticks(VARIATION_XTICKS)
            axis.set_xticklabels(tuple(map(str, VARIATION_XTICKS)))
            axis.minorticks_off()
            axis.set_title(title, fontsize=10, pad=5)
            axis.tick_params(labelsize=8, width=1.2, length=3)
            axis.spines["left"].set_linewidth(1.3)
            axis.spines["bottom"].set_linewidth(1.3)
            axis.grid(False)
    shared_upper = max(.2, math.ceil((max_sd + .025) * 10) / 10)
    for axis in plotted_axes:
        axis.set_ylim(0, shared_upper)
    handles = [
        Line2D([], [], color="black", linewidth=1.55, label="Fixed budget"),
        Line2D([], [], color=COLORS[1], marker="x", linestyle="None",
               markersize=5, label="Adaptive CS"),
        Line2D([], [], color=COLORS[2], marker="s", linestyle="None",
               markersize=4, label="Bethell et al."),
        Line2D([], [], color="#5B3F99", marker="D", linestyle="None",
               markersize=4, label="UncerTrans"),
    ]
    fig.supxlabel("Posterior samples per input (log scale)", fontsize=10, y=.085)
    fig.supylabel("Run-to-run accuracy SD (%)", fontsize=10, x=.015)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .02), fontsize=8.5, columnspacing=1.1,
               handletextpad=.4)
    fig.subplots_adjust(left=.095, right=.985, top=.95, bottom=.15,
                        wspace=.30, hspace=.43)
    fig.savefig(HERE / "accuracy_variation.pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(HERE / "accuracy_variation.png", dpi=240, transparent=True)
    plt.close(fig)


def run(*, cifar_only: bool | None = None) -> None:
    if cifar_only is None:
        cifar_only = not imagenet_repeats_complete()
        if cifar_only:
            print("ImageNet repeats incomplete; generating the paper's CIFAR panels with ImageNet pending")
    rows = {}
    curves = {}
    macros = {}
    largest_delta = 0.0
    largest_sd = 0.0
    largest_budget_gap = 0.0
    stability_wins = 0
    specs = SPECS[:2] if cifar_only else SPECS
    for benchmark, prefix, datasets in specs:
        latest = ROOT / "code" / "data" / benchmark / "latest_run.json"
        base = Path(json.loads(latest.read_text())["run_dir"]) / "full"
        summary = json.loads((base / "sample_efficiency" / "summary.json").read_text())
        for index, dataset in enumerate(datasets):
            with np.load(base / "sample_efficiency" / f"{dataset}.npz") as first:
                labels = first["labels"].copy()
                sources = first["source_indices"].copy()
                reference_class = first["reference_mean"].argmax(axis=1)
                adaptive_first = first["class_selection_estimates"].argmax(axis=1)
                fixed_first = first["class_selection_mean_matched_estimates"].argmax(axis=1)
                draws_first = first["class_selection_stop_times"].copy()
                statuses_first = first["class_selection_statuses"].copy()
            repeats_path = base.parent / "accuracy_repeats" / f"{dataset}.npz"
            with np.load(repeats_path) as fresh:
                if not np.array_equal(fresh["source_indices"], sources):
                    raise ValueError(f"{dataset}: source indices differ")
                if not np.array_equal(fresh["labels"], labels):
                    raise ValueError(f"{dataset}: labels differ")
                adaptive = np.vstack((adaptive_first[None], fresh["adaptive_predictions"]))
                fixed = np.vstack((fixed_first[None], fresh["fixed_predictions"]))
                draws = np.vstack((draws_first[None], fresh["adaptive_draws"]))
                statuses = np.vstack((statuses_first[None], fresh["adaptive_statuses"]))
                budget = int(fresh["fixed_budget"])
                seeds = [summary["config"]["seed"] + 1000 * index + 20, *fresh["seeds"].tolist()]
            expected = int(summary["datasets"][dataset]["tasks"]["class_selection"]["mean_matched_fixed_budget"])
            if budget != expected or adaptive.shape != (NUM_STREAMS, len(labels)):
                raise ValueError(f"{dataset}: budget or repeat count differs")
            seed_base = summary["config"]["seed"] + 1000 * index + 20
            if seeds != [seed_base + 100_000 * repeat for repeat in range(NUM_STREAMS)]:
                raise ValueError(f"{dataset}: repeat seeds differ")
            if np.any(draws < 1) or np.any(draws > summary["config"]["max_samples"]):
                raise ValueError(f"{dataset}: adaptive draw count outside the evaluation horizon")
            a = np.mean(adaptive == labels[None, :], axis=1) * 100
            f = np.mean(fixed == labels[None, :], axis=1) * 100
            d = a - f
            adaptive_changed_pct = float(np.mean(np.any(adaptive != adaptive[0], axis=0)) * 100)
            fixed_changed_pct = float(np.mean(np.any(fixed != fixed[0], axis=0)) * 100)
            row = {
                "dataset": dataset,
                "n_inputs": int(len(labels)),
                "seeds": seeds,
                "fixed_budget": budget,
                "adaptive_mean_draws_by_run": draws.mean(axis=1).tolist(),
                "adaptive_accuracy_pct_by_run": a.tolist(),
                "fixed_accuracy_pct_by_run": f.tolist(),
                "paired_accuracy_difference_pp_by_run": d.tolist(),
                "adaptive_accuracy_mean_pct": float(a.mean()),
                "adaptive_accuracy_sd_pp": float(a.std(ddof=1)),
                "fixed_accuracy_mean_pct": float(f.mean()),
                "fixed_accuracy_sd_pp": float(f.std(ddof=1)),
                "paired_difference_mean_pp": float(d.mean()),
                "paired_difference_sd_pp": float(d.std(ddof=1)),
                "adaptive_mean_draws": float(draws.mean()),
                "resolved_rate": float(np.mean(statuses == "stopped")),
                "adaptive_changed_prediction_pct": adaptive_changed_pct,
                "fixed_changed_prediction_pct": fixed_changed_pct,
                "reference_label_accuracy_pct": float(np.mean(reference_class == labels) * 100),
                "adaptive_reference_agreement_pct": float(np.mean(adaptive == reference_class[None, :]) * 100),
                "fixed_reference_agreement_pct": float(np.mean(fixed == reference_class[None, :]) * 100),
                "net_adaptive_correct_difference": int(np.sum(adaptive == labels[None, :]) - np.sum(fixed == labels[None, :])),
                "adaptive_fixed_disagreements": int(np.sum(adaptive != fixed)),
                "disagreements_on_adaptive_abstentions": int(np.sum((adaptive != fixed) & (statuses == "abstained"))),
            }
            rows[f"{benchmark}/{dataset}"] = row
            curves_path = base.parent / "accuracy_curves" / f"{dataset}.npz"
            with np.load(curves_path) as saved:
                data = {name: saved[name].copy() for name in saved.files}
            if not np.array_equal(data["source_indices"], sources):
                raise ValueError(f"{dataset}: accuracy-curve source indices differ")
            if not np.array_equal(data["labels"], labels):
                raise ValueError(f"{dataset}: accuracy-curve labels differ")
            if data["seeds"].tolist() != seeds:
                raise ValueError(f"{dataset}: accuracy-curve seeds differ")
            fixed_curve = data["fixed_accuracy_pct_by_run"]
            if fixed_curve.shape != (NUM_STREAMS, int(summary["config"]["max_samples"])):
                raise ValueError(f"{dataset}: fixed accuracy curve has wrong shape")
            if not np.allclose(fixed_curve[:, budget - 1], f, atol=1e-9):
                raise ValueError(f"{dataset}: fixed curve does not match the repeated table")
            for baseline in BASELINE_STYLES:
                if data[f"{baseline}_accuracy_pct_by_run"].shape != (NUM_STREAMS,):
                    raise ValueError(f"{dataset}: {baseline} repeat count differs")
                if data[f"{baseline}_mean_draws_by_run"].shape != (NUM_STREAMS,):
                    raise ValueError(f"{dataset}: {baseline} draw count differs")
                row[f"{baseline}_accuracy_mean_pct"] = float(data[f"{baseline}_accuracy_pct_by_run"].mean())
                row[f"{baseline}_accuracy_sd_pp"] = float(data[f"{baseline}_accuracy_pct_by_run"].std(ddof=1))
                row[f"{baseline}_mean_draws"] = float(data[f"{baseline}_mean_draws_by_run"].mean())
            curves[f"{benchmark}/{dataset}"] = data
            suffix = ("Clean", "C", "P")[index]
            stem = f"VisionRepeat{prefix}{suffix}"
            macros[stem + "Draws"] = f"{row['adaptive_mean_draws']:.1f}"
            macros[stem + "Budget"] = str(budget)
            macros[stem + "AdaptiveAccuracy"] = f"{a.mean():.2f}"
            macros[stem + "AdaptiveSD"] = f"{a.std(ddof=1):.2f}"
            macros[stem + "AdaptiveOneSDLower"] = f"{a.mean() - a.std(ddof=1):.2f}"
            macros[stem + "AdaptiveOneSDUpper"] = f"{a.mean() + a.std(ddof=1):.2f}"
            macros[stem + "FixedAccuracy"] = f"{f.mean():.2f}"
            macros[stem + "FixedSD"] = f"{f.std(ddof=1):.2f}"
            macros[stem + "FixedOneSDLower"] = f"{f.mean() - f.std(ddof=1):.2f}"
            macros[stem + "FixedOneSDUpper"] = f"{f.mean() + f.std(ddof=1):.2f}"
            if dataset == "cifar10_c" and benchmark == "cifar10":
                macros[stem + "FixedToAdaptiveSDRatio"] = f"{f.std(ddof=1) / a.std(ddof=1):.1f}"
            macros[stem + "Difference"] = f"{d.mean():+.2f}"
            macros[stem + "ReferenceAccuracy"] = f"{row['reference_label_accuracy_pct']:.2f}"
            macros[stem + "AdaptiveReferenceAgreement"] = f"{row['adaptive_reference_agreement_pct']:.2f}"
            macros[stem + "FixedReferenceAgreement"] = f"{row['fixed_reference_agreement_pct']:.2f}"
            macros[stem + "NetCorrectDifference"] = str(row["net_adaptive_correct_difference"])
            macros[stem + "FixedExtraCorrect"] = str(-row["net_adaptive_correct_difference"])
            macros[stem + "Disagreements"] = str(row["adaptive_fixed_disagreements"])
            macros[stem + "AbstainedDisagreements"] = str(row["disagreements_on_adaptive_abstentions"])
            largest_delta = max(largest_delta, abs(float(d.mean())))
            largest_sd = max(largest_sd, float(a.std(ddof=1)), float(f.std(ddof=1)))
            largest_budget_gap = max(largest_budget_gap, abs(float(draws.mean()) - budget))
            stability_wins += adaptive_changed_pct < fixed_changed_pct
    macros["VisionRepeatMaxAbsDifference"] = f"{largest_delta:.2f}"
    macros["VisionRepeatMaxSD"] = f"{largest_sd:.1f}"
    macros["VisionRepeatMaxBudgetGap"] = f"{largest_budget_gap:.1f}"
    macros["VisionRepeatStabilityWinsCount"] = str(stability_wins)
    macros["VisionRepeatStreams"] = str(len(seeds))
    macros["VisionRepeatDatasetCount"] = str(len(rows))
    if not cifar_only:
        image_p = rows["imagenet/imagenet_p"]
        macros["VisionRepeatImageNetPPairedEvaluations"] = f"{len(image_p['seeds']) * image_p['n_inputs']:,}"
    if cifar_only:
        macros["VisionRepeatPartialResults"] = "1"
    plot_accuracy(rows, curves, specs=specs, output_name="accuracy_repeats",
                  pending_imagenet=cifar_only)
    plot_cifar_c_precision(rows, curves)
    plot_accuracy_variation(rows, curves, pending_imagenet=cifar_only)
    (HERE / "accuracy_repeats_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (HERE / "accuracy_repeats_metrics.tex").write_text(
        "% Generated by generate_accuracy_repeats.py; do not edit.\n"
        + "\n".join(f"\\providecommand{{\\{key}}}{{{value}}}" for key, value in macros.items())
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-only", action="store_true",
                        help="render the six completed CIFAR panels while ImageNet runs")
    args = parser.parse_args()
    run(cifar_only=True if args.cifar_only else None)
