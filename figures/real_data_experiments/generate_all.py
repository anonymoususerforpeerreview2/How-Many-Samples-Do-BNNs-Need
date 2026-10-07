"""Generate the combined CIFAR-10/CIFAR-100/ImageNet paper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from bnn_demo_lib.plot_style import COLORS, configure_matplotlib  # noqa: E402

SPECS = {
    "cifar10": {
        "title": "CIFAR-10",
        "datasets": ("clean", "cifar10_c", "cifar10_p"),
        "labels": ("clean", "CIFAR-10-C", "CIFAR-10-P"),
        "prefix": "CifarTen",
    },
    "cifar100": {
        "title": "CIFAR-100",
        "datasets": ("clean", "cifar100_c", "cifar100_p"),
        "labels": ("clean", "CIFAR-100-C", "CIFAR-100-P"),
        "prefix": "Cifar",
    },
    "imagenet": {
        "title": "ImageNet-1K",
        "datasets": ("clean", "imagenet_c", "imagenet_p"),
        "labels": ("clean", "ImageNet-C", "ImageNet-P"),
        "prefix": "ImageNet",
    },
}
BASELINES = (("bethell", "s"), ("uncertrans", "D"))


def _run(
        benchmark: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    latest = PROJECT_ROOT / "code" / "data" / benchmark / "latest_run.json"
    run = Path(json.loads(latest.read_text(encoding="utf-8"))["run_dir"])
    full = run / "full"
    fit_path = full / "fit" / "summary.json"
    summary = json.loads((full / "sample_efficiency" / "summary.json").read_text(encoding="utf-8"))
    latency = json.loads((full / "latency" / "summary.json").read_text(encoding="utf-8"))
    schedule_path = full / "inspection_schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule["benchmark"] != benchmark or schedule["pool_size"] != 100:
        raise ValueError(f"{benchmark} has an invalid calibration artifact")
    if sum(schedule["pool_composition"].values()) != schedule["pool_size"]:
        raise ValueError(f"{benchmark} calibration pool composition is inconsistent")
    def source_keys(values):
        return {tuple(item) if isinstance(item, list) else item for item in values}
    for dataset in ("clean", f"{benchmark}_c", f"{benchmark}_p"):
        with np.load(full / "sample_efficiency" / f"{dataset}.npz") as saved:
            evaluated = source_keys(saved["source_indices"].tolist())
        held_out = source_keys(schedule["source_indices"][dataset])
        if evaluated & held_out or len(held_out) != schedule["pool_composition"][dataset]:
            raise ValueError(f"{benchmark}/{dataset}: calibration pool overlaps evaluation")
    for task in ("class_selection", "threshold", "distribution_accuracy"):
        expected = list(schedule["tasks"][task]["inspection_times"])
        for dataset in ("clean", f"{benchmark}_c", f"{benchmark}_p"):
            actual = summary["datasets"][dataset]["tasks"][task]["inspection_times"]
            if actual != expected:
                raise ValueError(f"{benchmark}/{dataset}/{task}: schedule differs")
    for payload in (summary, latency):
        config = payload["config"]
        if (config.get("early_abstention_window") != 30
                or config.get("early_abstention_late_offsets") != [20, 10, 5]
                or 470 not in schedule["tasks"]["class_selection"]["inspection_times"]):
            raise ValueError(f"{benchmark} results do not use the final-30-draw protocol")
    if benchmark == "imagenet":
        # Older fetched runs include metadata for the discarded dropout search.
        summary.pop("dropout_validation", None)
        for payload in (summary, latency):
            for key in (
                "dropout_candidates",
                "dropout_validation_inputs",
                "dropout_validation_samples",
                "max_bma_accuracy_drop",
                "max_draw_accuracy_drop",
            ):
                payload["config"].pop(key, None)
    return (
        run,
        summary,
        latency,
        json.loads(fit_path.read_text(encoding="utf-8")) if fit_path.is_file() else None,
    )


def _arrays(run: Path, dataset: str) -> dict[str, np.ndarray]:
    with np.load(run / "full" / "sample_efficiency" / f"{dataset}.npz") as payload:
        return {key: payload[key] for key in payload.files}


def _metric(task: str, arrays: dict[str, np.ndarray], threshold: float) -> float:
    certified = arrays[f"{task}_statuses"] == "stopped"
    if not certified.any():
        return float("nan")
    if task == "class_selection":
        target = arrays["reference_mean"].argmax(axis=1)
        estimate = arrays["class_selection_estimates"].argmax(axis=1)
        return float(np.mean((estimate == target)[certified]))
    if task == "distribution_accuracy":
        errors = np.max(
            np.abs(arrays["distribution_accuracy_estimates"] - arrays["reference_mean"]),
            axis=1,
        )
        return float(np.mean(errors[certified]))
    target = arrays["reference_mean"].max(axis=1) >= threshold
    decision = arrays["threshold_decisions"].astype(bool)
    return float(np.mean((decision == target)[certified]))


def _all_input_metric(task: str, arrays: dict[str, np.ndarray], threshold: float) -> float:
    """Score each final posterior mean, including unresolved inputs."""
    reference = arrays["reference_mean"]
    estimates = arrays[f"{task}_estimates"]
    if task == "class_selection":
        return float(np.mean(estimates.argmax(axis=1) == reference.argmax(axis=1)))
    if task == "distribution_accuracy":
        return float(np.mean(np.max(np.abs(estimates - reference), axis=1)))
    target = reference.max(axis=1) >= threshold
    return float(np.mean((estimates.max(axis=1) >= threshold) == target))


def _quality_matched_budget(task: str, curve: np.ndarray, metric: float) -> int:
    """First fixed budget with quality at least as good as the all-input CS point."""
    comparable = curve <= metric + 1e-12 if task == "distribution_accuracy" else curve >= metric - 1e-12
    matches = np.flatnonzero(comparable)
    if not len(matches):
        raise ValueError(f"fixed curve never matches all-input {task} quality")
    return int(matches[0] + 1)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def _write_macros(path: Path, values: dict[str, str]) -> None:
    lines = ["% Generated from local runs; do not edit."]
    vision_started = False
    for key, value in values.items():
        if key.startswith("Vision") and not vision_started:
            lines.extend(["", "% Numeric cells for benchmark_summary.tex and runtime_all.tex."])
            vision_started = True
        lines.append(f"\\providecommand{{\\{key}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _density(stop_times: np.ndarray, horizon: int) -> np.ndarray:
    """Evaluate a Gaussian KDE of stopping times on the sample grid."""
    values = np.asarray(stop_times, dtype=float)
    grid = np.arange(1, horizon + 1, dtype=float)
    if values.size == 0:
        return np.zeros_like(grid)
    if np.all(values == values[0]):
        # gaussian_kde cannot estimate a covariance matrix from a single
        # distinct value. Use its limiting Gaussian with one grid-unit scale.
        scale = 1.0
        return np.exp(-0.5 * ((grid - values[0]) / scale) ** 2) / (
            scale * np.sqrt(2 * np.pi)
        )
    return gaussian_kde(values, bw_method=0.05)(grid)


TASK_PANELS = (
    ("class_selection", r"$\operatorname{Pr}\left(\hat{c}_l=c^*\right)$"),
    ("distribution_accuracy", r"$\left\|\hat{\boldsymbol{\mu}}_t-\boldsymbol{\mu}\right\|_{\infty}$"),
    ("threshold", r"$\text{Pr(correct threshold decision)}$"),
)


def _legend() -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D([], [], color="black", linewidth=1.55),
        Line2D([], [], color=COLORS[1], linewidth=1.55),
        Line2D([], [], color=COLORS[2], linewidth=1.55),
        Line2D([], [], color=".25", marker="o", linestyle="None", markersize=6,
               markerfacecolor="none"),
        Line2D([], [], color=".25", marker="x", linestyle="None", markersize=7),
        Line2D([], [], color=".25", marker="s", linestyle="None"),
        Line2D([], [], color=".25", marker="D", linestyle="None"),
    ]
    labels = [
        "Clean", "C (severity 5)", "P (final frame)",
        "Adaptive CS (all inputs)", "Adaptive CS (resolved)", "Bethell et al.", "UncerTrans",
    ]
    return handles, labels


def _certification_text(rates: list[float]) -> str:
    return "Resolved (Clean / C / P)\n" + " / ".join(
        f"{100 * rate:.0f}" for rate in rates
    ) + "%"


def _benchmark_row_labels(fig: plt.Figure, axes: np.ndarray) -> None:
    """Label benchmark rows in the margin instead of covering their curves."""
    for row, spec in enumerate(SPECS.values()):
        bounds = axes[row, 0].get_position()
        fig.text(
            .025, (bounds.y0 + bounds.y1) / 2, spec["title"],
            ha="center", va="center", rotation=90, fontsize=10,
            fontweight="bold",
        )


def _vision_panel_style(axis: plt.Axes) -> None:
    """Match the panel strokes and text sizes in accuracy_variation.pdf."""
    axis.tick_params(labelsize=8, width=1.2, length=3)
    axis.spines["left"].set_linewidth(1.3)
    axis.spines["bottom"].set_linewidth(1.3)
    axis.grid(False)


def _quality_figure(runs: dict[str, dict[str, Any]]) -> None:
    """All three quality-versus-budget panels, grouped by benchmark."""
    first_sample = 5
    last_sample = min(int(run["summary"]["config"]["max_samples"]) for run in runs.values())
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 7.7))
    colors = ("black", COLORS[1], COLORS[2])
    titles = ("(a) Class selection", "(b) Distribution accuracy", "(c) Threshold decision")
    for row, (benchmark, spec) in enumerate(SPECS.items()):
        if benchmark not in runs:
            for axis in axes[row]:
                axis.set_axis_off()
                axis.text(.5, .5, "ImageNet results pending", ha="center", va="center",
                          transform=axis.transAxes, fontsize=9, color="0.45")
            continue
        summary = runs[benchmark]["summary"]
        for column, ((task_name, ylabel), title) in enumerate(zip(TASK_PANELS, titles)):
            axis = axes[row, column]
            panel_values: list[float] = []
            certified_rates: list[float] = []
            for dataset, label, color in zip(spec["datasets"], spec["labels"], colors):
                arrays = runs[benchmark]["arrays"][dataset]
                curve = arrays[f"{task_name}_fixed_metric"]
                budgets = np.arange(first_sample, min(last_sample, len(curve)) + 1)
                task = summary["datasets"][dataset]["tasks"][task_name]
                certified_rates.append(task["adaptive_certified_rate"])
                metric = _metric(task_name, arrays, summary["config"]["probability_threshold"])
                all_metric = _all_input_metric(task_name, arrays, summary["config"]["probability_threshold"])
                shown_curve = curve[budgets - 1]
                shown_metric = metric
                shown_all_metric = all_metric
                panel_values.extend(shown_curve.tolist())
                if first_sample <= task["adaptive_mean_samples"] <= last_sample:
                    panel_values.extend((shown_metric, shown_all_metric))
                axis.plot(budgets, shown_curve, color=color, linewidth=1.55, label=label)
                axis.scatter(
                    task["adaptive_mean_samples"], shown_all_metric,
                    facecolors="none", edgecolors=color,
                    marker="o", s=34, linewidths=1.2, zorder=4,
                )
                axis.scatter(
                    task["adaptive_mean_samples"], shown_metric, color=color,
                    marker="x", s=34, linewidths=1.3, zorder=5,
                )
                for key, marker in BASELINES:
                    baseline = summary["datasets"][dataset].get("baselines", {}).get(key)
                    if baseline is None:
                        continue
                    if first_sample <= baseline["mean_samples"] <= last_sample:
                        panel_values.append(float(baseline["metrics"][task_name]))
                    axis.scatter(
                        baseline["mean_samples"], baseline["metrics"][task_name],
                        color=color, marker=marker, s=34,
                        edgecolors="white", linewidths=.5,
                    )
            if row == 1:
                axis.set_ylabel(ylabel, fontsize=10)
            axis.set_xscale("log")
            axis.set_xlim(first_sample, last_sample)
            axis.set_xticks((10, 20, 50, 100, 200, 500))
            axis.set_xticklabels(("10", "20", "50", "100", "200", "500"))
            axis.minorticks_off()
            margin = .08 * (max(panel_values) - min(panel_values))
            axis.set_ylim(max(0, min(panel_values) - margin),
                          min(1.01, max(panel_values) + margin)
                          if task_name != "distribution_accuracy"
                          else max(panel_values) + margin)
            if row == 0:
                axis.set_title(title, fontsize=10, pad=5)
            axis.text(
                .98, .96 if task_name == "distribution_accuracy" else .04,
                _certification_text(certified_rates),
                va="top" if task_name == "distribution_accuracy" else "bottom",
                ha="right", transform=axis.transAxes, fontsize=8,
                bbox={"facecolor": "white", "edgecolor": ".7", "alpha": .9, "pad": 2},
            )
            _vision_panel_style(axis)
    handles, labels = _legend()
    fig.supxlabel("Posterior samples per input (log scale)", fontsize=10, y=.085)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .02), fontsize=8.5, columnspacing=1.1,
               handletextpad=.4)
    fig.subplots_adjust(left=.095, right=.985, top=.95, bottom=.15,
                        wspace=.30, hspace=.43)
    _benchmark_row_labels(fig, axes)
    fig.savefig(
        HERE / "vision_quality.pdf", transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _allocation_figure(runs: dict[str, dict[str, Any]]) -> None:
    """Stopping-time distributions for all tasks and benchmarks."""
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 7.7))
    colors = ("black", COLORS[1], COLORS[2])
    titles = ("(a) Class selection", "(b) Distribution accuracy", "(c) Threshold decision")
    for row, (benchmark, spec) in enumerate(SPECS.items()):
        if benchmark not in runs:
            for axis in axes[row]:
                axis.set_axis_off()
                axis.text(.5, .5, "ImageNet results pending", ha="center", va="center",
                          transform=axis.transAxes, fontsize=9, color="0.45")
            continue
        summary = runs[benchmark]["summary"]
        horizon = int(summary["config"]["max_samples"])
        for column, ((task_name, _), title) in enumerate(zip(TASK_PANELS, titles)):
            axis = axes[row, column]
            certified_rates: list[float] = []
            for dataset, label, color in zip(spec["datasets"], spec["labels"], colors):
                arrays = runs[benchmark]["arrays"][dataset]
                task = summary["datasets"][dataset]["tasks"][task_name]
                certified_rates.append(task["adaptive_certified_rate"])
                axis.plot(
                    np.arange(1, horizon + 1),
                    _density(arrays[f"{task_name}_stop_times"], horizon),
                    color=color, linewidth=1.55, label=label,
                )
            axis.set_xlim(-0.05 * horizon, horizon + (horizon * 0.05))
            if row == 1 and column == 1:
                axis.set_ylabel("Density", fontsize=10)
            if row == 0:
                axis.set_title(title, fontsize=10, pad=5)
            axis.text(
                .98, .96, _certification_text(certified_rates),
                va="top", ha="right", transform=axis.transAxes, fontsize=8,
                bbox={"facecolor": "white", "edgecolor": ".7", "alpha": .9, "pad": 2},
            )
            _vision_panel_style(axis)
    handles = [
        Line2D([], [], color="black", linewidth=1.55),
        Line2D([], [], color=COLORS[1], linewidth=1.55),
        Line2D([], [], color=COLORS[2], linewidth=1.55),
    ]
    labels = ["Clean", "C (severity 5)", "P (final frame)"]
    fig.supxlabel("Posterior samples per input", fontsize=10, y=.085)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .02), fontsize=8.5, columnspacing=1.1,
               handletextpad=.4)
    fig.subplots_adjust(left=.095, right=.985, top=.95, bottom=.15,
                        wspace=.30, hspace=.43)
    _benchmark_row_labels(fig, axes)
    fig.savefig(
        HERE / "vision_allocation.pdf", transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _cifar100_early_abstention_comparisons(
        run: dict[str, Any], task: str = "class_selection",
) -> dict[str, Any]:
    """Compare CIFAR-100 early abstention with the capped outcome."""
    horizon = int(run["summary"]["config"]["max_samples"])
    comparisons: dict[str, dict[str, Any]] = {}
    for dataset in SPECS["cifar100"]["datasets"]:
        arrays = run["arrays"][dataset]
        early_statuses = arrays[f"{task}_statuses"]
        early_times = arrays[f"{task}_stop_times"]
        if task == "class_selection":
            baseline_statuses = arrays["class_selection_no_early_statuses"]
            baseline_times = arrays["class_selection_no_early_stop_times"]
        else:
            # Safe futility means an early abstention cannot become a decision
            # by the cap. These runs did not save a separate no-early replay.
            baseline_statuses = np.where(early_statuses == "stopped", "stopped", "timeout")
            baseline_times = np.where(early_statuses == "stopped", early_times, horizon)
        if not np.all(np.isin(baseline_statuses, ("stopped", "timeout"))):
            raise ValueError(f"CIFAR-100 {dataset}/{task} baseline has unknown statuses")
        baseline_resolved = baseline_statuses == "stopped"
        if (
            not np.array_equal(early_statuses[baseline_resolved], baseline_statuses[baseline_resolved])
            or not np.array_equal(early_times[baseline_resolved], baseline_times[baseline_resolved])
            or np.any(baseline_statuses[early_statuses != "stopped"] != "timeout")
        ):
            raise ValueError(
                f"CIFAR-100 {dataset}/{task} early-abstention replay loses a "
                "baseline certification"
            )
        if not np.all(np.isin(early_statuses, ("stopped", "abstained", "timeout"))):
            raise ValueError(
                f"CIFAR-100 {dataset}/{task} early-abstention replay has unknown statuses"
            )
        if (np.any(early_times < 1) or np.any(early_times > horizon)
                or np.any(early_times[early_statuses == "timeout"] != horizon)):
            raise ValueError(f"CIFAR-100 {dataset}/{task} has invalid stopping times")

        inconclusive = ~baseline_resolved
        early_abstained = inconclusive & (early_statuses == "abstained")
        # The baseline contains only seven-character status names (``stopped``
        # and ``timeout``); widen before inserting ``abstained``.
        outcome_statuses = baseline_statuses.astype("<U10", copy=True)
        outcome_times = baseline_times.copy()
        outcome_statuses[inconclusive] = "timeout"
        outcome_times[inconclusive] = horizon
        outcome_statuses[early_abstained] = "abstained"
        outcome_times[early_abstained] = early_times[early_abstained]
        savings = np.zeros(int(np.sum(inconclusive)), dtype=int)
        early_inconclusive = early_statuses[inconclusive] == "abstained"
        savings[early_inconclusive] = (
                horizon
                - early_times[inconclusive][early_inconclusive]
        )
        comparisons[dataset] = {
            "statuses": outcome_statuses,
            "stop_times": outcome_times,
            "savings": savings,
            "resolved_count": int(np.sum(baseline_resolved)),
            "timeout_count": int(np.sum(inconclusive) - np.sum(early_abstained)),
            "abstained_count": int(np.sum(early_abstained)),
            "inconclusive_count": int(np.sum(inconclusive)),
        }
    return {
        "source_run": run["run"],
        "task": task,
        "horizon": horizon,
        "datasets": comparisons,
    }


def _class_allocation_kde_figure(
        class_comparison: dict[str, Any],
        horizon: int,
        output_name: str,
        *,
        show_quantiles: bool = True,
) -> dict[str, Any]:
    """Class-selection stopping-time density for a clean CIFAR dataset."""
    stop_times = class_comparison["stop_times"].astype(int)
    statuses = class_comparison["statuses"]
    if (len(stop_times) != len(statuses)
            or not np.all(np.isin(statuses, ("stopped", "abstained", "timeout")))
            or np.any(stop_times < 1) or np.any(stop_times > horizon)):
        raise ValueError(f"{output_name}: invalid class-selection outcomes")
    total = len(statuses)
    grid = np.arange(1, horizon + 1)
    quantiles = np.quantile(stop_times, (.25, .50, .75), method="linear")
    outcome_groups = (
        ("stopped", "Resolved"),
        ("abstained", "Abstained early"),
        ("timeout", f"Unresolved at {horizon}"),
    )

    fig, axis = plt.subplots(figsize=(3.4, 2.55))
    curve = _density(stop_times, horizon)
    axis.plot(grid, curve, color="black", linewidth=1.7, zorder=3)
    axis.fill_between(grid, curve, color="black", alpha=.08, zorder=2)
    if show_quantiles:
        quantile_groups: dict[float, list[int]] = {}
        for index, value in enumerate(quantiles, start=1):
            quantile_groups.setdefault(float(value), []).append(index)
        quantile_labels: list[str] = []
        for value, indices in quantile_groups.items():
            axis.axvline(
                value, color=COLORS[0], linestyle=(0, (2.2, 2.2)),
                linewidth=1.15, alpha=.85, zorder=1,
            )
            quantile_labels.append(
                "$" + "=".join(rf"Q_{index}" for index in indices)
                + rf"={value:g}$"
            )
        axis.text(
            .45, .62, "\n".join(quantile_labels),
            color=COLORS[0], fontsize=9.5, linespacing=1.25,
            ha="left", va="top", transform=axis.transAxes,
        )

    outcome_labels = [
        f"{label}: {100 * np.mean(statuses == status):.1f}%"
        for status, label in outcome_groups
    ]
    axis.text(
        .45,
        # .35,
        .97,
        "\n".join(outcome_labels),
        transform=axis.transAxes, ha="left", va="top",
        fontsize=9, linespacing=1.35, color="0.15",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": "0.65", "linewidth": 0.8},
    )
    axis.set(
        xlabel="Posterior samples per input",
        ylabel="Density",
        xlim=(-0.05 * horizon, horizon + (horizon * 0.05))
        # axis.set(xlim=(-0.05 * horizon, horizon + (horizon * 0.05)), ylabel="Density")
    )
    fig.tight_layout(pad=.45)
    if output_name == "cifar10_c_allocation_kde":
        axis.set_title("Stopping-time density", fontsize=17.5, pad=6)
        # Reserve space above the axis without changing its size beside the
        # accuracy panel in Figure 8.
        old_height = fig.get_figheight()
        x, y, width, height = axis.get_position().bounds
        fig.set_figheight(old_height + .3)
        scale = old_height / fig.get_figheight()
        axis.set_position((x, y * scale, width, height * scale))
    fig.savefig(
        HERE / f"{output_name}.pdf", transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return {
        "resolved_count": int(np.sum(statuses == "stopped")),
        "abstained_count": int(np.sum(statuses == "abstained")),
        "timeout_count": int(np.sum(statuses == "timeout")),
        "total": total,
        "quantiles": quantiles,
    }


def _cifar100_early_abstention_figure(comparison: dict[str, Any]) -> None:
    """Appendix savings diagnostic for clean and shifted CIFAR-100."""
    labels = ("clean", "CIFAR-100-C", "CIFAR-100-P")
    datasets = SPECS["cifar100"]["datasets"]
    max_saved = max(
        int(np.max(comparison["datasets"][dataset]["savings"]))
        for dataset in datasets
    )
    bins = np.arange(-.5, max_saved + 1.5, 1)
    fig, axes = plt.subplots(1, 3, figsize=(8.25, 2.65), sharex=True, sharey=True)
    distribution = comparison["task"] == "distribution_accuracy"
    for axis, dataset, label in zip(axes, datasets, labels):
        values = comparison["datasets"][dataset]
        savings = values["savings"]
        axis.hist(savings, bins=bins, color="black", edgecolor="white")
        axis.axvline(
            np.mean(savings), color=COLORS[1], linestyle="--", linewidth=1.4,
            label=f"Mean = {np.mean(savings):.1f}",
        )
        axis.set(
            title=label,
            xlabel="Posterior samples saved",
            xlim=(-.5, max_saved + .5),
            xticks=np.arange(0, max_saved + 1, 2),
        )
        axis.text(
            .96 if distribution else .04, .79,
            f"{values['abstained_count']}/{values['inconclusive_count']} detected early",
            transform=axis.transAxes, fontsize=8,
            ha="right" if distribution else "left",
        )
        axis.legend(frameon=False, fontsize=8.0,
                    loc="upper right" if distribution else "upper left")
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(
        HERE / (
            "cifar100_early_stopping_evaluation.pdf"
            if comparison["task"] == "class_selection"
            else f"cifar100_{comparison['task']}_early_stopping_evaluation.pdf"
        ), transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _additional_early_macros(comparisons: dict[str, Any]) -> dict[str, str]:
    macros: dict[str, str] = {}
    for task, comparison in comparisons.items():
        task_suffix = "Threshold" if task == "threshold" else "Distribution"
        for dataset, dataset_suffix in zip(
                SPECS["cifar100"]["datasets"], ("", "C", "P")):
            values = comparison["datasets"][dataset]
            prefix = f"Cifar{dataset_suffix}{task_suffix}Early"
            macros.update({
                prefix + "InconclusiveCount": str(values["inconclusive_count"]),
                prefix + "AbstainedCount": str(values["abstained_count"]),
                prefix + "MeanSaved": f"{np.mean(values['savings']):.1f}",
            })
    return macros


def _additional_early_summary(comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        task: {
            "source_run": str(comparison["source_run"]),
            "datasets": {
                dataset: {
                    "resolved_count": values["resolved_count"],
                    "timeout_count": values["timeout_count"],
                    "abstained_count": values["abstained_count"],
                    "inconclusive_count": values["inconclusive_count"],
                    "mean_samples_saved": float(np.mean(values["savings"])),
                }
                for dataset, values in comparison["datasets"].items()
            },
        }
        for task, comparison in comparisons.items()
    }


def generate_cifar100_early_only() -> None:
    """Update the companion figures when another benchmark run is unavailable."""
    configure_matplotlib()
    latest = PROJECT_ROOT / "code" / "data" / "cifar100" / "latest_run.json"
    run_dir = Path(json.loads(latest.read_text(encoding="utf-8"))["run_dir"])
    summary = json.loads(
        (run_dir / "full" / "sample_efficiency" / "summary.json").read_text(encoding="utf-8")
    )
    run = {
        "run": run_dir,
        "summary": summary,
        "arrays": {
            dataset: _arrays(run_dir, dataset)
            for dataset in SPECS["cifar100"]["datasets"]
        },
    }
    comparisons = {
        task: _cifar100_early_abstention_comparisons(run, task)
        for task in ("threshold", "distribution_accuracy")
    }
    for comparison in comparisons.values():
        _cifar100_early_abstention_figure(comparison)
    macros = _additional_early_macros(comparisons)
    metrics_path = HERE / "metrics_all.tex"
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    prefixes = tuple(f"\\providecommand{{\\{key}}}" for key in macros)
    lines = [line for line in lines if not line.startswith(prefixes)]
    lines.extend(f"\\providecommand{{\\{key}}}{{{value}}}" for key, value in macros.items())
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = HERE / "summary_all.json"
    paper_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paper_summary["cifar100_other_early_abstention"] = _additional_early_summary(comparisons)
    summary_path.write_text(json.dumps(paper_summary, indent=2) + "\n", encoding="utf-8")


def _table_metrics(runs: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Only table values are generated; captions, headings and rows live in TeX."""
    metrics: dict[str, str] = {}
    for benchmark in runs:
        spec = SPECS[benchmark]
        summary = runs[benchmark]["summary"]
        schedule = json.loads((runs[benchmark]["run"] / "full" / "inspection_schedule.json").read_text())
        for task, suffix in (("class_selection", "Class"),
                             ("threshold", "Threshold"),
                             ("distribution_accuracy", "Distribution")):
            times = schedule["tasks"][task]["inspection_times"]
            metrics[f"Vision{spec['prefix']}{suffix}Inspections"] = (
                ",\\allowbreak ".join(str(time) for time in times)
            )
        threshold = summary["config"]["probability_threshold"]
        for dataset, suffix in zip(spec["datasets"], ("Clean", "C", "P")):
            values = summary["datasets"][dataset]
            arrays = runs[benchmark]["arrays"][dataset]
            class_task = values["tasks"]["class_selection"]
            dist_task = values["tasks"]["distribution_accuracy"]
            threshold_task = values["tasks"]["threshold"]
            class_agreement = _metric("class_selection", arrays, threshold)
            threshold_agreement = _metric("threshold", arrays, threshold)
            prefix = f"Vision{spec['prefix']}{suffix}"
            metrics.update({
                prefix + "ReferenceAccuracyPct": _pct(values["reference_accuracy"]),
                prefix + "ClassAgreementPct": _pct(class_agreement),
                prefix + "ClassErrorPct": _pct(1 - class_agreement),
                prefix + "ClassCertifiedPct": _pct(class_task["adaptive_certified_rate"]),
                prefix + "ClassDraws": f"{class_task['adaptive_mean_samples']:.1f}",
                prefix + "DistributionError": f"{_metric('distribution_accuracy', arrays, threshold):.4f}",
                prefix + "DistributionCertifiedPct": _pct(dist_task["adaptive_certified_rate"]),
                prefix + "DistributionDraws": f"{dist_task['adaptive_mean_samples']:.1f}",
                prefix + "ThresholdAgreementPct": _pct(threshold_agreement),
                prefix + "ThresholdErrorPct": _pct(1 - threshold_agreement),
                prefix + "ThresholdCertifiedPct": _pct(threshold_task["adaptive_certified_rate"]),
                prefix + "ThresholdDraws": f"{threshold_task['adaptive_mean_samples']:.1f}",
            })
    runtime_counts = {
        runs[benchmark]["latency"]["num_runtime_inputs"] for benchmark in runs
    }
    if len(runtime_counts) != 1:
        raise ValueError("runtime input counts must match across benchmarks")
    runtime_count = runtime_counts.pop()
    for benchmark in runs:
        summary_config = runs[benchmark]["summary"]["config"]
        if runtime_count != summary_config["num_inputs"]:
            raise ValueError(f"{benchmark} latency does not match sample efficiency")
    metrics["VisionRuntimeInputCount"] = f"{runtime_count:,}"
    sample_run = next(iter(runs.values()))["run"]
    calibration = json.loads((sample_run / "full" / "inspection_schedule.json").read_text())
    metrics["VisionCalibrationPoolSize"] = str(calibration["pool_size"])
    metrics["VisionCalibrationCleanCount"] = str(calibration["pool_composition"]["clean"])
    shifted_counts = [count for name, count in calibration["pool_composition"].items()
                      if name != "clean"]
    if len(set(shifted_counts)) != 1:
        raise ValueError("C and P calibration counts differ")
    metrics["VisionCalibrationShiftedCount"] = str(shifted_counts[0])
    metrics["VisionCalibrationPercentileStep"] = str(calibration["quantiles_pct"][1] - calibration["quantiles_pct"][0])
    metrics["VisionCalibrationLastPercentile"] = str(calibration["quantiles_pct"][-1])
    first_batch = json.loads((sample_run / "full" / "batch_latency" / "summary.json").read_text())
    metrics["VisionBatchSize"] = str(first_batch["batch_size"])
    metrics["VisionAdaptiveComparisonCount"] = str(3 * len(runs))
    overheads: dict[str, list[float]] = {"Cifar": [], "ImageNet": []}
    adaptive_wins = 0
    for benchmark in runs:
        spec = SPECS[benchmark]
        latency = runs[benchmark]["latency"]
        fixed_path = runs[benchmark]["run"] / "full" / "fixed_budget_latency" / "summary.json"
        fixed_latency = json.loads(fixed_path.read_text(encoding="utf-8"))
        batch_path = runs[benchmark]["run"] / "full" / "batch_latency" / "summary.json"
        batch_latency = json.loads(batch_path.read_text(encoding="utf-8"))
        if (batch_latency["batch_size"] != 16
                or batch_latency["num_inputs"] != runtime_count
                or batch_latency["gpu"] != latency["gpu"]):
            raise ValueError(f"{benchmark} batched timing protocol does not match")
        if (
            fixed_latency["num_runtime_inputs"] != runtime_count
            or fixed_latency["gpu"] != latency["gpu"]
            or fixed_latency["batch_size"] != latency["batch_size"]
            or fixed_latency["config"]["runtime_repeats"]
            != latency["config"]["runtime_repeats"]
        ):
            raise ValueError(f"{benchmark} fixed-budget timing protocol does not match")
        for task, suffix in (
            ("class_selection", "Class"),
            ("threshold", "Threshold"),
            ("distribution_accuracy", "Distribution"),
        ):
            result = latency["tasks"][task]
            cs_pct = 100 * result["cs_seconds"] / result["inference_seconds"]
            overheads["ImageNet" if benchmark == "imagenet" else "Cifar"].append(cs_pct)
            prefix = f"Vision{spec['prefix']}Runtime{suffix}"
            metrics.update({
                prefix + "Draws": f"{result['mean_draws']:.1f}",
                prefix + "InferenceSeconds": f"{result['inference_seconds']:.3f}",
                prefix + "CSSeconds": f"{result['cs_seconds']:.3f}",
                prefix + "TotalSeconds": f"{result['total_seconds']:.3f}",
                prefix + "CSOverheadPct": f"{cs_pct:.1f}",
            })
            arrays = runs[benchmark]["arrays"]["clean"]
            target = _all_input_metric(
                task, arrays,
                runs[benchmark]["summary"]["config"]["probability_threshold"],
            )
            curve = arrays[f"{task}_fixed_metric"]
            matched = _quality_matched_budget(task, curve, target)
            fixed_result = fixed_latency["tasks"][task]
            if fixed_result["draws"] != matched:
                raise ValueError(f"{benchmark} {task} fixed latency uses wrong budget")
            adaptive_seconds = result["total_seconds"]
            fixed_seconds = fixed_result["inference_seconds"]
            adaptive_wins += int(adaptive_seconds < fixed_seconds)
            metrics.update({
                prefix + "AllInputMetric": f"{target:.6f}",
                prefix + "FixedMetric": f"{curve[matched - 1]:.6f}",
                prefix + "FixedDraws": str(matched),
                prefix + "FixedInferenceSeconds": f"{fixed_result['inference_seconds']:.3f}",
                prefix + "AdaptiveWins": str(int(adaptive_seconds < fixed_seconds)),
                prefix + "Speedup": f"{max(adaptive_seconds, fixed_seconds) / min(adaptive_seconds, fixed_seconds):.2f}",
            })
            batched = batch_latency["tasks"][task]
            if (batched["fixed_draws"] != matched
                    or not np.isclose(batched["mean_draws"], result["mean_draws"])):
                raise ValueError(f"{benchmark} {task} batched comparison differs")
            batch_prefix = f"Vision{spec['prefix']}Batch{suffix}"
            metrics.update({
                batch_prefix + "Draws": f"{batched['mean_draws']:.1f}",
                batch_prefix + "InferenceSeconds": f"{batched['inference_seconds_per_input']:.3f}",
                batch_prefix + "CSSeconds": f"{batched['cs_seconds_per_input']:.3f}",
                batch_prefix + "TotalSeconds": f"{batched['adaptive_seconds_per_input']:.3f}",
                batch_prefix + "FixedDraws": str(matched),
                batch_prefix + "FixedSeconds": f"{batched['fixed_seconds_per_input']:.3f}",
                batch_prefix + "Ratio": f"{batched['fixed_over_adaptive_ratio']:.2f}",
            })
    for benchmark, values in overheads.items():
        if values:
            metrics[f"Vision{benchmark}RuntimeCSOverheadMinPct"] = f"{min(values):.1f}"
            metrics[f"Vision{benchmark}RuntimeCSOverheadMaxPct"] = f"{max(values):.1f}"
    metrics["VisionAdaptiveWinsCount"] = str(adaptive_wins)
    return metrics


def main(*, cifar_only: bool = False) -> None:
    configure_matplotlib()
    runs: dict[str, dict[str, Any]] = {}
    macros: dict[str, str] = {}
    benchmarks = ("cifar10", "cifar100") if cifar_only else tuple(SPECS)
    for benchmark in benchmarks:
        spec = SPECS[benchmark]
        run, summary, latency, fit = _run(benchmark)
        arrays = {dataset: _arrays(run, dataset) for dataset in spec["datasets"]}
        runs[benchmark] = {
            "run": run,
            "summary": summary,
            "latency": latency,
            "fit": fit,
            "arrays": arrays,
        }
        threshold = summary["config"]["probability_threshold"]
        clean = summary["datasets"]["clean"]
        prefix = spec["prefix"]
        macros[prefix + "ReferenceAccuracyPct"] = _pct(clean["reference_accuracy"])
        for task, suffix in (
                ("class_selection", "Class"),
                ("distribution_accuracy", "Distribution"),
                ("threshold", "Threshold"),
        ):
            result = clean["tasks"][task]
            metric = _metric(task, arrays["clean"], threshold)
            macros[prefix + suffix + ("Error" if task == "distribution_accuracy" else "AgreementPct")] = (
                f"{metric:.4f}" if task == "distribution_accuracy" else _pct(metric)
            )
            macros[prefix + suffix + "CertifiedPct"] = _pct(result["adaptive_certified_rate"])
            macros[prefix + suffix + "MeanSamples"] = f"{result['adaptive_mean_samples']:.1f}"
        if benchmark == "cifar100":
            class_task = clean["tasks"]["class_selection"]
            macros["CifarClassUnresolvedPct"] = _pct(
                1.0 - class_task["adaptive_certified_rate"]
            )
            macros["CifarClassQOneSamples"] = f"{class_task['adaptive_q25_samples']:.0f}"
            macros["CifarClassMedianSamples"] = f"{class_task['adaptive_median_samples']:.0f}"
            macros["CifarClassQThreeSamples"] = f"{class_task['adaptive_q75_samples']:.0f}"
            for dataset, suffix in (("cifar100_c", "C"), ("cifar100_p", "P")):
                shifted = summary["datasets"][dataset]["tasks"]["class_selection"]
                macros[f"Cifar{suffix}AgreementPct"] = _pct(
                    _metric("class_selection", arrays[dataset], threshold)
                )
                macros[f"Cifar{suffix}MeanSamples"] = f"{shifted['adaptive_mean_samples']:.1f}"
                macros[f"Cifar{suffix}CertifiedPct"] = _pct(
                    shifted["adaptive_certified_rate"]
                )
            assert fit is not None
            prior = float(fit["posterior"]["prior_precision"])
            exponent = int(np.floor(np.log10(prior)))
            coefficient = prior / (10 ** exponent)
            macros["CifarPriorPrecision"] = (
                f"{coefficient:g}\\times10^{{{exponent}}}"
            )
            curvature = float(fit["fit_timing"]["curvature_seconds"])
            macros["CifarCurvatureSeconds"] = f"{curvature:.1f}"
            macros["CifarCurvatureMinutes"] = f"{curvature / 60:.1f}"
        if benchmark == "imagenet":
            macros["ImageNetDropoutProbability"] = f"{summary['posterior']['dropout_probability']:.2f}"
    if cifar_only:
        from experiments.imagenet.config import ImageNetConfig
        macros["ImageNetDropoutProbability"] = f"{ImageNetConfig.dropout_probability:.2f}"
        macros["VisionPartialResults"] = "1"
    cifar100_early = _cifar100_early_abstention_comparisons(runs["cifar100"])
    clean_early = cifar100_early["datasets"]["clean"]
    total = len(clean_early["statuses"])
    macros.update(
        {
            "CifarClassResolvedCount": str(clean_early["resolved_count"]),
            "CifarClassResolvedPct": _pct(clean_early["resolved_count"] / total),
            "CifarClassTimeoutCount": str(clean_early["timeout_count"]),
            "CifarClassTimeoutPct": _pct(clean_early["timeout_count"] / total),
            "CifarClassAbstainedCount": str(clean_early["abstained_count"]),
            "CifarClassAbstainedPct": _pct(clean_early["abstained_count"] / total),
            "CifarEarlyInconclusiveCount": str(clean_early["inconclusive_count"]),
            "CifarEarlyMeanSaved": f"{np.mean(clean_early['savings']):.1f}",
            "CifarEarlyMedianSaved": f"{np.median(clean_early['savings']):.0f}",
        }
    )
    for dataset, suffix in (("cifar100_c", "C"), ("cifar100_p", "P")):
        values = cifar100_early["datasets"][dataset]
        macros.update(
            {
                f"Cifar{suffix}EarlyInconclusiveCount": str(values["inconclusive_count"]),
                f"Cifar{suffix}EarlyAbstainedCount": str(values["abstained_count"]),
                f"Cifar{suffix}EarlyStillCappedCount": str(values["timeout_count"]),
                f"Cifar{suffix}EarlyMeanSaved": f"{np.mean(values['savings']):.1f}",
                f"Cifar{suffix}EarlyMedianSaved": f"{np.median(values['savings']):.0f}",
            }
        )
    other_early = {
        task: _cifar100_early_abstention_comparisons(runs["cifar100"], task)
        for task in ("threshold", "distribution_accuracy")
    }
    _quality_figure(runs)
    _allocation_figure(runs)
    class_allocation = _class_allocation_kde_figure(
        clean_early, cifar100_early["horizon"], "cifar100_allocation_kde",
    )
    if class_allocation["total"] != total:
        raise ValueError("ClassSelection allocation summary has inconsistent size")
    macros.update(
        {
            "CifarClassQOneSamples": f"{class_allocation['quantiles'][0]:.0f}",
            "CifarClassMedianSamples": f"{class_allocation['quantiles'][1]:.0f}",
            "CifarClassQThreeSamples": f"{class_allocation['quantiles'][2]:.0f}",
        }
    )
    cifar10_c = runs["cifar10"]["arrays"]["cifar10_c"]
    cifar10_c_allocation = _class_allocation_kde_figure(
        {
            "stop_times": cifar10_c["class_selection_stop_times"],
            "statuses": cifar10_c["class_selection_statuses"],
        },
        int(runs["cifar10"]["summary"]["config"]["max_samples"]),
        "cifar10_c_allocation_kde",
        show_quantiles=False,
    )
    macros.update(
        {
            "CifarTenCClassMedianSamples": f"{cifar10_c_allocation['quantiles'][1]:.0f}",
        }
    )
    _cifar100_early_abstention_figure(cifar100_early)
    for comparison in other_early.values():
        _cifar100_early_abstention_figure(comparison)
    macros.update(_table_metrics(runs))
    macros.update(_additional_early_macros(other_early))
    _write_macros(HERE / "metrics_all.tex", macros)
    (HERE / "summary_all.json").write_text(
        json.dumps(
            {
                "metric_policy": (
                    "certified CS quality uses resolved inputs; all-input crosses "
                    "score final posterior means for every input; quality-matched "
                    "fixed budgets use the all-input metric"
                ),
                **{
                    key: {
                        "source_run": str(value["run"]),
                        "summary": value["summary"],
                        "latency": value["latency"],
                        "fixed_budget_latency": json.loads(
                            (value["run"] / "full" / "fixed_budget_latency" / "summary.json").read_text()
                        ),
                        "batch_latency": json.loads(
                            (value["run"] / "full" / "batch_latency" / "summary.json").read_text()
                        ),
                        "inspection_calibration": json.loads(
                            (value["run"] / "full" / "inspection_schedule.json").read_text()
                        ),
                        "all_input_metrics": {
                            dataset: {
                                task: _all_input_metric(
                                    task, value["arrays"][dataset],
                                    value["summary"]["config"]["probability_threshold"],
                                )
                                for task, _ in TASK_PANELS
                            }
                            for dataset in SPECS[key]["datasets"]
                        },
                        "quality_matched_fixed_clean": {
                            task: {
                                "budget": _quality_matched_budget(
                                    task,
                                    value["arrays"]["clean"][f"{task}_fixed_metric"],
                                    _all_input_metric(
                                        task, value["arrays"]["clean"],
                                        value["summary"]["config"]["probability_threshold"],
                                    ),
                                ),
                                "target_all_input_metric": _all_input_metric(
                                    task, value["arrays"]["clean"],
                                    value["summary"]["config"]["probability_threshold"],
                                ),
                            }
                            for task, _ in TASK_PANELS
                        },
                    }
                    for key, value in runs.items()
                },
                "cifar100_early_abstention": {
                    "source_run": str(cifar100_early["source_run"]),
                    "datasets": {
                        dataset: {
                            "resolved_count": values["resolved_count"],
                            "timeout_count": values["timeout_count"],
                            "abstained_count": values["abstained_count"],
                            "inconclusive_count": values["inconclusive_count"],
                            "mean_samples_saved": float(np.mean(values["savings"])),
                            "median_samples_saved": float(np.median(values["savings"])),
                        }
                        for dataset, values in cifar100_early["datasets"].items()
                    },
                },
                "cifar100_other_early_abstention": _additional_early_summary(other_early),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Generated CIFAR-only artifacts with ImageNet placeholders" if cifar_only
          else "Generated combined CIFAR-10, CIFAR-100, and ImageNet artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-only", action="store_true")
    parser.add_argument("--cifar-early-only", action="store_true")
    args = parser.parse_args()
    if args.cifar_early_only:
        generate_cifar100_early_only()
    else:
        main(cifar_only=args.cifar_only)
