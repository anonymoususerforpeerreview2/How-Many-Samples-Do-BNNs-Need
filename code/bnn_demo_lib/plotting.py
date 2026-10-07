"""Publication-ready figures for the BNN demonstration."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .settings import CSBuilder
from .data import sample_dirichlet_around, sample_predictive_around, select_example_indices
from .plot_style import COLORS, PAPER_GUIDE_LINE_WIDTH, PAPER_LINE_WIDTH, configure_matplotlib

if TYPE_CHECKING:
    from .cifar100 import Cifar100Result


def _class_colors(num_classes: int) -> list:
    """Return distinct colors while preserving the paper's three-class palette."""
    if num_classes <= len(COLORS):
        return list(COLORS[:num_classes])
    return list(plt.get_cmap("tab20")(np.linspace(0.0, 1.0, num_classes)))


def _draw_simplex(ax: plt.Axes) -> np.ndarray:
    """Draw a 3-class simplex and return its Cartesian vertex coordinates."""
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)))
    closed = np.vstack((vertices, vertices[0]))
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=PAPER_LINE_WIDTH)
    labels = ("class 1", "class 2", "class 3")
    offsets = ((-0.03, -0.055), (1.03, -0.055), (0.5, 0.91))
    alignments = ("right", "left", "center")
    for label, position, alignment in zip(labels, offsets, alignments):
        ax.text(
            *position,
            label,
            ha=alignment,
            va="center" if alignment == "center" else "top",
            fontsize=11,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.10, 0.96)
    ax.axis("off")
    return vertices


@dataclass(frozen=True)
class FigureWriter:
    """Save figures and optionally keep them open for interactive display."""

    output_dir: Path
    formats: Sequence[str]
    dpi: int
    show: bool = False

    def save(self, figure: plt.Figure, name: str) -> None:
        for extension in self.formats:
            save_options = {
                "dpi": self.dpi,
                "transparent": True,
            }
            if extension.lower() == "pdf":
                # Avoid timestamp-only Git diffs when figures are regenerated.
                save_options["metadata"] = {
                    "CreationDate": None,
                    "ModDate": None,
                }
            figure.savefig(
                self.output_dir / f"{name}.{extension}",
                **save_options,
            )
        if not self.show:
            plt.close(figure)

    def display(self) -> None:
        """Display saved figures when requested, then release their resources."""
        if self.show:
            plt.show()
            plt.close("all")


def plot_stopping_histogram(
    stop_times: np.ndarray,
    statuses: np.ndarray,
    histogram_bins: int,
    writer: FigureWriter,
) -> None:
    stop_times = np.asarray(stop_times)
    statuses = np.asarray(statuses)
    if stop_times.ndim != 1 or statuses.shape != stop_times.shape:
        raise ValueError("stop_times and statuses must be one-dimensional and aligned.")

    resolved = statuses == "stopped"
    early_abstention = statuses == "abstained"
    empty = statuses == "empty"
    unresolved = (statuses == "timeout") | empty
    recognized = resolved | early_abstention | unresolved
    if not np.all(recognized):
        unknown = sorted(set(statuses[~recognized].tolist()))
        raise ValueError(f"Unknown adaptive prediction statuses: {unknown}.")

    unresolved_label = r"Unresolved at $T_{\max}$"
    if np.any(empty):
        unresolved_label += " / empty CS"
    outcome_groups = (
        (resolved, "Resolved", COLORS[0], None),
        (
            early_abstention,
            "Early abstention (provably unresolved)",
            COLORS[2],
            "///",
        ),
        (unresolved, unresolved_label, COLORS[1], "..."),
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    _, bin_edges = np.histogram(stop_times, bins=histogram_bins)
    bin_widths = np.diff(bin_edges)
    bottoms = np.zeros(histogram_bins, dtype=float)
    for mask, label, color, hatch in outcome_groups:
        counts, _ = np.histogram(stop_times[mask], bins=bin_edges)
        ax.bar(
            bin_edges[:-1],
            counts,
            width=bin_widths,
            bottom=bottoms,
            align="edge",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
            label=label,
        )
        bottoms += counts
    ax.set(
        xlabel="MC samples used (stopping time)",
        ylabel="Count",
        title="Adaptive stopping times",
    )
    ax.legend(title="Outcome within each bin", frameon=False)
    fig.tight_layout()
    writer.save(fig, "poster_stop_hist")


def plot_simplex_distributions(
    means: np.ndarray,
    concentrations: np.ndarray,
    is_hard: np.ndarray,
    seed: int,
    num_samples: int,
    writer: FigureWriter,
) -> None:
    """Plot easy and ambiguous predictive draws on the 2-simplex."""
    hard, easy = select_example_indices(is_hard)
    rng = np.random.default_rng(seed + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))

    for ax, index, label in zip(
        axes,
        (easy, hard),
        ("Easy input", "Ambiguous input"),
    ):
        vertices = _draw_simplex(ax)
        samples = rng.dirichlet(
            np.maximum(1e-8, concentrations[index] * means[index]),
            size=num_samples,
        )
        locations = samples @ vertices
        ax.scatter(
            locations[:, 0],
            locations[:, 1],
            s=10,
            alpha=0.16,
            color=COLORS[0],
            edgecolors="none",
        )
        mean_location = means[index] @ vertices
        ax.scatter(
            *mean_location,
            marker="*",
            s=190,
            color=COLORS[1],
            edgecolor="white",
            linewidth=0.8,
            label=r"true $\mu$",
        )
        ax.set_title(rf"{label} ($\kappa={concentrations[index]:.1f}$)")
        ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        r"Predictive samples on $\Delta^2$",
        y=0.99,
    )
    fig.tight_layout()
    writer.save(fig, "poster_simplex_predictive")


def plot_simplex_overlay(
    means: np.ndarray,
    concentrations: np.ndarray,
    is_hard: np.ndarray,
    seed: int,
    num_samples: int,
    writer: FigureWriter,
) -> None:
    """Overlay representative easy and ambiguous predictions on one simplex."""
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    _plot_simplex_overlay_axis(
        ax, means, concentrations, is_hard, seed, num_samples
    )
    ax.set_title(r"Predictive samples on $\Delta^2$")
    fig.tight_layout()
    writer.save(fig, "paper_simplex_overlay")


def _stop_label(rule: str, stop_t: int, outcome: str) -> str:
    """Format a compact stopping-time legend entry."""
    labels = {
        "class": "Class selection",
        "distribution": "Distribution accuracy",
        "threshold": "Threshold selection",
        "box": "Confidence box",
    }
    return f"{labels.get(rule, rule)} ($t={stop_t}$)"


def _plot_cs_axis(
    ax: plt.Axes,
    mean: np.ndarray,
    concentration: float,
    cs_builder: CSBuilder,
    seed: int,
    max_samples: int,
    title: str,
    *,
    show_ylabel: bool,
    show_legend: bool,
    class_label_start: int = 1,
    stopping_times: Optional[dict[str, Optional[int]]] = None,
    distribution_epsilon: float = 0.05,
    probability_threshold: float = 0.7,
    component_means: Optional[np.ndarray] = None,
    component_weights: Optional[np.ndarray] = None,
    samples: Optional[np.ndarray] = None,
) -> None:
    """Plot class means and confidence bands for one deterministic stream."""
    rng = np.random.default_rng(seed)
    tracker = cs_builder()
    mean_history, lower_history, upper_history = [], [], []
    class_stop = None
    for sample_count in range(1, max_samples + 1):
        if samples is not None:
            draw = samples[sample_count - 1]
        elif component_means is not None and component_weights is not None:
            draw = sample_predictive_around(
                component_means, component_weights, concentration, rng
            )
        else:
            draw = sample_dirichlet_around(mean, concentration, rng)
        result = tracker.update(draw)
        mean_history.append(result.mean[0].cpu().numpy())
        lower_history.append(result.lower[0].cpu().numpy())
        upper_history.append(result.upper[0].cpu().numpy())
        if class_stop is None and bool(result.certified[0]):
            class_stop = sample_count

    sample_counts = np.arange(1, max_samples + 1)
    mean_history = np.asarray(mean_history)
    lower_history = np.asarray(lower_history)
    upper_history = np.asarray(upper_history)
    for class_index, color in enumerate(_class_colors(len(mean))):
        ax.plot(
            sample_counts,
            mean_history[:, class_index],
            color=color,
            label=f"class {class_index + class_label_start}",
            zorder=3,
        )
        ax.fill_between(
            sample_counts,
            lower_history[:, class_index],
            upper_history[:, class_index],
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

    if stopping_times is None:
        if class_stop is not None:
            ax.axvline(class_stop, color="black", linestyle=":", linewidth=1.35)
            ax.plot([], [], color="black", linestyle=":", linewidth=1.35,
                    label=_stop_label("class", int(class_stop), ""))
    else:
        class_t = stopping_times.get("class")
        if class_t is not None:
            index = int(np.clip(class_t - 1, 0, max_samples - 1))
            leader = int(np.argmax(mean_history[index]))
            competitors = upper_history[index].copy()
            competitors[leader] = -np.inf
            competitor = int(np.argmax(competitors))
            lower_leader = lower_history[index, leader]
            upper_competitor = upper_history[index, competitor]
            midpoint = 0.5 * (lower_leader + upper_competitor)
            ax.scatter(
                class_t,
                midpoint,
                marker="o",
                s=30,
                linewidths=0.7,
                color="black",
                edgecolors="white",
                zorder=6,
            )
            ax.annotate(
                f"Class selection ($t={class_t}$)",
                xy=(class_t, midpoint),
                xytext=(class_t + 34, midpoint + (0.08 if np.max(mean) > 0.7 else 0.01)),
                fontsize=8.5,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
                zorder=7,
            )

        threshold_t = stopping_times.get("threshold")
        ax.hlines(
            probability_threshold, 0, max_samples,
            color="black",
            linestyle=(0, (2.0, 2.0)),
            linewidth=0.75,
            alpha=0.7,
            zorder=0,
        )
        ax.text(
            0.985,
            probability_threshold + 0.015,
            rf"$q={probability_threshold:g}$",
            transform=ax.get_yaxis_transform(),
            color="black",
            fontsize=8.5,
            ha="right",
            va="bottom",
        )
        if threshold_t is not None:
            index = int(np.clip(threshold_t - 1, 0, max_samples - 1))
            ax.scatter(
                threshold_t,
                probability_threshold,
                marker="o",
                s=30,
                color="black",
                edgecolor="white",
                linewidth=0.7,
                zorder=6,
            )
            ax.annotate(
                f"Threshold decision ($t={threshold_t}$)",
                xy=(threshold_t, probability_threshold),
                xytext=(threshold_t + 35, 0.77 if np.max(mean) > 0.7 else 0.82),
                fontsize=8.5,
                color="black",
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
                zorder=7,
            )

        distribution_t = stopping_times.get("distribution")
        if distribution_t is not None:
            index = int(np.clip(distribution_t - 1, 0, max_samples - 1))
            for class_index in range(len(mean)):
                estimate = mean_history[index, class_index]
                ax.errorbar(
                    distribution_t,
                    estimate,
                    yerr=distribution_epsilon,
                    fmt="o",
                    markersize=2.8,
                    color="black",
                    ecolor="black",
                    elinewidth=0.8,
                    capsize=2.5,
                    capthick=0.8,
                    zorder=6,
                )
            easy_panel = np.max(mean) > 0.7
            target_class = int(np.argmin(mean_history[index]) if easy_panel else np.argmax(mean_history[index]))
            target_y = mean_history[index, target_class] + distribution_epsilon
            ax.annotate(
                rf"Distribution accuracy ($t={distribution_t}$, $\varepsilon={distribution_epsilon:g}$)",
                xy=(distribution_t, target_y),
                xytext=(140 if easy_panel else 145, 0.18 if easy_panel else 0.91),
                fontsize=8.5,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
                zorder=7,
            )

    ax.set(
        xlabel="Posterior samples",
        ylabel="Class probability" if show_ylabel else "",
        ylim=(-0.05, 1.0),
        xlim=(-0.05 * max_samples, max_samples),
        title=title,
    )
    if show_legend:
        ax.legend(ncol=2, fontsize=9.5, frameon=False, loc="upper right")


def plot_toy_sequence_overview(
    means: np.ndarray,
    concentrations: np.ndarray,
    is_hard: np.ndarray,
    cs_builder: CSBuilder,
    seed: int,
    num_samples: int,
    max_samples: int,
    writer: FigureWriter,
    stopping_times: Optional[dict[str, dict[str, Optional[int]]]] = None,
    component_means: Optional[np.ndarray] = None,
    component_weights: Optional[np.ndarray] = None,
    samples: Optional[np.ndarray] = None,
    example_indices: Optional[tuple[int, int]] = None,
    distribution_epsilon: float = 0.05,
    probability_threshold: float = 0.7,
) -> None:
    """Create the paper's aligned simplex/easy/ambiguous three-panel figure."""
    hard, easy = example_indices or select_example_indices(is_hard)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 3.15),
        gridspec_kw={"width_ratios": [1.0, 1.35, 1.35]},
    )
    _plot_simplex_overlay_axis(
        axes[0], means, concentrations, is_hard, seed, num_samples,
        component_means=component_means,
        component_weights=component_weights,
        example_indices=(hard, easy),
    )
    axes[0].set_title("(a) Samples from two inputs")
    _plot_cs_axis(
        axes[1],
        means[easy],
        concentrations[easy],
        cs_builder,
        seed + 30_002,
        max_samples,
        "(b) Easy input",
        show_ylabel=True,
        show_legend=False,
        stopping_times=None if stopping_times is None else stopping_times.get("easy"),
        distribution_epsilon=distribution_epsilon,
        probability_threshold=probability_threshold,
        component_means=None if component_means is None else component_means[easy],
        component_weights=None if component_weights is None else component_weights[easy],
        samples=None if samples is None else samples[:, easy],
    )
    _plot_cs_axis(
        axes[2],
        means[hard],
        concentrations[hard],
        cs_builder,
        seed + 30_001,
        max_samples,
        "(c) Ambiguous input",
        show_ylabel=False,
        show_legend=False,
        stopping_times=None if stopping_times is None else stopping_times.get("hard"),
        distribution_epsilon=distribution_epsilon,
        probability_threshold=probability_threshold,
        component_means=None if component_means is None else component_means[hard],
        component_weights=None if component_weights is None else component_weights[hard],
        samples=None if samples is None else samples[:, hard],
    )
    # Keep the shared class legend outside the data region; stopping rules are
    # labeled directly at the confidence-bound feature that certifies them.
    legend_handles, legend_labels = axes[2].get_legend_handles_labels()
    axes[2].legend(
        legend_handles,
        legend_labels,
        ncol=1,
        fontsize=10.5,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.subplots_adjust(left=0.04, right=0.865, bottom=0.18, top=0.88, wspace=0.28)
    writer.save(fig, "paper_toy_sequences")


def plot_epistemic_uncertainty(
    config,
    seed: int,
    max_samples: int,
    writer: FigureWriter,
) -> dict[str, dict[str, float | int | None]]:
    """Show posterior spread and an anytime estimate of epistemic uncertainty."""
    import adaptive_cs as acs
    from scipy.special import digamma

    def entropy(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-12, 1.0)
        return -(clipped * np.log(clipped)).sum(axis=-1)

    mean = np.asarray([0.78, 0.16, 0.06])
    cases = (("Stable posterior", 80.0), ("Disagreeing posterior", 3.0))
    num_display_samples = min(350, max_samples)
    trajectories = []
    summary = {}
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)))

    for case_index, (label, concentration) in enumerate(cases):
        rng = np.random.default_rng(seed + 10 * case_index)
        draws = rng.dirichlet(concentration * mean, size=max_samples)
        tracker = acs.Uncertainty(
            component="epistemic",
            stopping="accuracy",
            epsilon=0.05,
            max_samples=max_samples,
            config=config,
            stop_on_resolve=False,
        )
        components = ("epistemic", "aleatoric", "total")
        estimates = {component: [] for component in components}
        lowers = {component: [] for component in components}
        uppers = {component: [] for component in components}
        for sample_count, draw in enumerate(draws, start=1):
            result = tracker.update(draw)
            for component in components:
                interval = getattr(result, component)
                estimates[component].append(float(interval.estimate[0]))
                lowers[component].append(float(interval.lower[0]))
                uppers[component].append(float(interval.upper[0]))

        alpha = concentration * mean
        alpha_0 = float(alpha.sum())
        expected_entropy = float(
            digamma(alpha_0 + 1.0)
            - np.dot(alpha / alpha_0, digamma(alpha + 1.0))
        )
        references = {
            "epistemic": float(entropy(mean) - expected_entropy),
            "aleatoric": expected_entropy,
            "total": float(entropy(mean)),
        }
        trajectories.append(
            (label, concentration, draws,
             {component: np.asarray(values) for component, values in estimates.items()},
             {component: np.asarray(values) for component, values in lowers.items()},
             {component: np.asarray(values) for component, values in uppers.items()},
             references)
        )
        summary[label.lower().replace(" ", "_")] = {
            "concentration": concentration,
            "reference_epistemic_nats": references["epistemic"],
            "reference_aleatoric_nats": references["aleatoric"],
            "reference_total_nats": references["total"],
            "final_estimate_nats": float(estimates["epistemic"][-1]),
        }

    # One horizontal composite: the first two panels show the posterior draws,
    # followed by one trajectory panel for each uncertainty component.
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(15.8, 3.6),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.7, 1.7, 1.7)},
    )
    component_labels = {
        "epistemic": "Epistemic uncertainty",
        "aleatoric": "Aleatoric uncertainty",
        "total": "Total uncertainty",
    }
    for column, (label, concentration, draws, estimates, lowers, uppers,
                 references) in enumerate(trajectories):
        top = axes[column]
        _draw_simplex(top)
        locations = draws[:num_display_samples] @ vertices
        top.scatter(locations[:, 0], locations[:, 1], s=8, alpha=0.18,
                    color=COLORS[column], edgecolors="none")
        top.scatter(*(mean @ vertices), marker="*", s=150,
                    color=COLORS[1], edgecolor="black", linewidth=0.7)
        top.set_title(f"{label}\n($\\kappa={concentration:g}$)")

    sample_counts = np.arange(1, max_samples + 1)
    for panel_index, component in enumerate(("epistemic", "aleatoric", "total"), start=2):
        axis = axes[panel_index]
        axis.set_title(component_labels[component])
        for column, (label, concentration, draws, estimates, lowers, uppers,
                     references) in enumerate(trajectories):
            axis.fill_between(sample_counts, lowers[component], uppers[component],
                              color=COLORS[column], alpha=0.18, linewidth=0,
                              label="95% confidence interval" if column == 0 else None)
            axis.plot(sample_counts, estimates[component], color=COLORS[column],
                      label=f"{label} estimate")
            axis.axhline(references[component], color="black", linestyle=":",
                         linewidth=1.2,
                         label="Analytic reference" if column == 0 else None)
            if column == 0 or not np.isclose(
                references[component], trajectories[0][6][component]
            ):
                axis.annotate(
                    f"analytic reference = {references[component]:.3f} nats",
                    xy=(max_samples, references[component]), xytext=(-4, 5),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=COLORS[column],
                )

        axis.set(xlabel="Posterior samples", xlim=(0, max_samples + 1),
                 ylim=(0.0, 1.0), ylabel="nats")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        fontsize=8.5,
        frameon=False,
        ncol=4,
        columnspacing=0.8,
        handlelength=1.8,
    )
    fig.suptitle(
        "Same mean prediction, different model disagreement", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    writer.save(fig, "paper_epistemic_uncertainty")
    return summary


def _plot_simplex_overlay_axis(
    ax: plt.Axes,
    means: np.ndarray,
    concentrations: np.ndarray,
    is_hard: np.ndarray,
    seed: int,
    num_samples: int,
    *,
    component_means: Optional[np.ndarray] = None,
    component_weights: Optional[np.ndarray] = None,
    example_indices: Optional[tuple[int, int]] = None,
) -> None:
    hard, easy = example_indices or select_example_indices(is_hard)
    rng = np.random.default_rng(seed + 11)
    vertices = _draw_simplex(ax)
    for index, label, color in (
        (easy, "Easy input", COLORS[0]),
        (hard, "Ambiguous input", COLORS[2]),
    ):
        if component_means is None or component_weights is None:
            samples = rng.dirichlet(
                np.maximum(1e-8, concentrations[index] * means[index]),
                size=num_samples,
            )
        else:
            samples = np.asarray([
                sample_predictive_around(
                    component_means[index], component_weights[index], concentrations[index], rng
                )
                for _ in range(num_samples)
            ])
        locations = samples @ vertices
        ax.scatter(
            locations[:, 0],
            locations[:, 1],
            s=9,
            alpha=0.18,
            color=color,
            edgecolors="none",
            label=label,
        )
        mean_location = means[index] @ vertices
        ax.scatter(
            *mean_location,
            marker="*",
            s=150,
            color=color,
            edgecolor="black",
            linewidth=PAPER_GUIDE_LINE_WIDTH,
        )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=12,
        columnspacing=0.8,
        handlelength=0.8,
        handletextpad=0.25,
        scatterpoints=1,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(0.95)
        handle.set_sizes([24])


def plot_predictive_marginals(
    means: np.ndarray,
    concentrations: np.ndarray,
    is_hard: np.ndarray,
    seed: int,
    num_samples: int,
    writer: FigureWriter,
) -> None:
    """Plot coordinatewise marginals of easy and ambiguous predictions."""
    hard, easy = select_example_indices(is_hard)
    rng = np.random.default_rng(seed + 2)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    for ax, index, label in zip(
        axes,
        (easy, hard),
        ("Easy input", "Ambiguous input"),
    ):
        samples = rng.dirichlet(
            np.maximum(1e-8, concentrations[index] * means[index]),
            size=num_samples,
        )
        parts = ax.violinplot(
            samples,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("black")
            body.set_edgecolor("black")
            body.set_alpha(0.55)
        parts["cmeans"].set_color(COLORS[1])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Class probability")
        ax.set_title(rf"{label} ($\kappa={concentrations[index]:.1f}$)")

    axes[-1].set_xlabel("Class index")
    axes[-1].set_xticks(np.arange(1, means.shape[1] + 1), range(means.shape[1]))
    fig.tight_layout()
    writer.save(fig, "poster_predictive_dists")


def plot_cs_shrinking_example(
    mean: np.ndarray,
    concentration: float,
    cs_builder: CSBuilder,
    seed: int,
    max_samples: int,
    title: str,
    output_name: str,
    writer: FigureWriter,
    class_label_start: int = 1,
    only_show=False,
) -> None:
    """Plot running class means and confidence bounds for one input."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _plot_cs_axis(
        ax,
        mean,
        concentration,
        cs_builder,
        seed,
        max_samples,
        title,
        show_ylabel=True,
        show_legend=True,
        class_label_start=class_label_start,
    )
    fig.tight_layout()
    if only_show:
        plt.show()
    else:
        writer.save(fig, output_name)


def _result_outcome(result, criterion: str):
    """Return the paper-facing outcome label once a rule certifies."""
    if not bool(result.certified[0]):
        return None
    if criterion == "class":
        return "selected"
    if criterion == "distribution":
        return "accurate"
    return "reached" if bool(result.decision[0]) else "not reached"

def plot_cs_bbox_example(
    mean: np.ndarray,
    concentration: float,
    bounding_box_builder: CSBuilder,
    seed: int,
    max_samples: int,
    title: str,
    output_name: str,
    writer: FigureWriter,
    class_label_start: int = 0,
    only_show: bool = False,
    stopping_criterion: str = "class",
    distribution_epsilon: float = 0.05,
    probability_threshold: float = 0.7,
) -> None:
    """Plot posterior means and bounding-box CS bounds for one input."""
    rng = np.random.default_rng(seed)
    tracker = bounding_box_builder()
    num_classes = int(len(mean))
    means, lowers, uppers = [], [], []
    stop = None

    for sample_count in range(1, max_samples + 1):
        draw = sample_dirichlet_around(mean, concentration, rng)
        result = tracker.update(draw)
        means.append(result.mean[0].cpu().numpy())
        lowers.append(result.lower[0].cpu().numpy())
        uppers.append(result.upper[0].cpu().numpy())
        if stop is None:
            outcome = _result_outcome(result, stopping_criterion)
            if outcome is not None:
                stop = (sample_count, outcome)

    sample_counts = np.arange(1, max_samples + 1)
    mean_history = np.asarray(means)
    lower_history = np.asarray(lowers)
    upper_history = np.asarray(uppers)

    fig, ax = plt.subplots(figsize=(8.1, 4.5))
    class_handles = []
    for class_index, color in enumerate(_class_colors(num_classes)):
        (mean_line,) = ax.plot(
            sample_counts,
            mean_history[:, class_index],
            color=color,
            label=f"class {class_index + class_label_start}",
            zorder=3,
        )
        class_handles.append(mean_line)
        ax.fill_between(
            sample_counts,
            lower_history[:, class_index],
            upper_history[:, class_index],
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

    method_handle = Line2D(
        [0], [0], color="0.25", linewidth=8, alpha=0.14,
        label="Bounding-box projection",
    )
    stop_handles = []
    if stop is not None:
        stop_t, outcome = stop
        ax.axvline(stop_t, color="black", linestyle=":", linewidth=1.5)
        stop_handles.append(
            Line2D(
                [0], [0], color="black", linestyle=":", linewidth=1.5,
                label=_stop_label("box", stop_t, outcome),
            )
        )
    ax.set(
        xlabel="Posterior samples",
        ylabel="Class probability",
        ylim=(0.0, 1.0),
    )
    class_legend_rows = math.ceil(num_classes / 5)
    ax.set_title(title, pad=30 + 12 * (class_legend_rows - 1))
    class_legend = ax.legend(
        handles=class_handles,
        ncol=min(num_classes, 5),
        fontsize=10.5,
        frameon=False,
        loc="upper center",
    )
    ax.add_artist(class_legend)
    ax.legend(
        handles=[method_handle, *stop_handles],
        ncol=2,
        fontsize=10.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout()
    fig.subplots_adjust(
        bottom=0.32,
        top=0.86 - 0.03 * (class_legend_rows - 1),
    )
    if only_show:
        plt.show()
    else:
        writer.save(fig, output_name)
