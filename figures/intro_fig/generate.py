"""Generate the introductory fixed-budget figures.

Run this file without arguments, for example:

    python figures/intro_fig/generate.py

The main-text figure shows only the wrong-label rate of a fixed posterior-
sample budget.  A supplementary figure combines representative running
posterior-predictive means with the corresponding simplex distributions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from bnn_demo_lib.plotting import (
    COLORS as PAPER_COLORS,
    PAPER_LINE_WIDTH,
    configure_matplotlib as configure_paper_style,
)


SCRIPT_DIR = Path(__file__).resolve().parent
N_RUNS = 10_000
FIXED_SAMPLES = 30
ERROR_HORIZON = 100
SEED = 20260915

EASY_MEAN = np.asarray((0.78, 0.16, 0.06))
# Keep the easy input well separated on average, but leave a small finite-
# sample error at the very shortest budgets so the error curve is visible.
EASY_CONCENTRATION = 5.0
RUNNING_HORIZON = 100

# Two concentrated modes model posterior disagreement between classes 1 and 2,
# while the mixture mean still has a statistically identifiable leading class.
HARD_MEAN = np.asarray((0.50, 0.40, 0.10))
HARD_COMPONENT_MEANS = np.asarray(((0.75, 0.15, 0.10), (0.15, 0.75, 0.10)))
HARD_COMPONENT_CONCENTRATION = 100.0
HARD_COMPONENT_ONE_WEIGHT = (HARD_MEAN[0] - HARD_COMPONENT_MEANS[1, 0]) / (
    HARD_COMPONENT_MEANS[0, 0] - HARD_COMPONENT_MEANS[1, 0]
)

CLASS_COLORS = PAPER_COLORS


def configure_matplotlib() -> None:
    configure_paper_style()


def sample_dirichlet_batch(
    means: np.ndarray, concentration: float, rng: np.random.Generator
) -> np.ndarray:
    gamma = rng.gamma(shape=concentration * means, scale=1.0)
    return gamma / gamma.sum(axis=1, keepdims=True)


def sample_batch(case: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if case == "easy":
        means = np.repeat(EASY_MEAN[None, :], size, axis=0)
        return sample_dirichlet_batch(means, EASY_CONCENTRATION, rng)
    choose_first = rng.random(size) < HARD_COMPONENT_ONE_WEIGHT
    means = HARD_COMPONENT_MEANS[np.where(choose_first, 0, 1)]
    return sample_dirichlet_batch(means, HARD_COMPONENT_CONCENTRATION, rng)


def generate_streams(case: str, n_runs: int, horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    streams = np.empty((n_runs, horizon, 3), dtype=np.float64)
    for time in range(horizon):
        streams[:, time, :] = sample_batch(case, n_runs, rng)
    return streams


def true_mean(case: str) -> np.ndarray:
    return EASY_MEAN if case == "easy" else HARD_MEAN


def choose_illustrative_stream(case: str, horizon: int) -> np.ndarray:
    """Choose a deterministic stream with an informative early trajectory."""
    for candidate_seed in range(10_000):
        stream = generate_streams(case, 1, horizon, SEED + candidate_seed)[0]
        running = np.cumsum(stream, axis=0) / np.arange(1, horizon + 1)[:, None]
        target_at_30 = 1 if case == "hard" else 0
        if np.argmax(running[FIXED_SAMPLES - 1]) != target_at_30:
            continue
        if np.ptp(running[: min(10, horizon)], axis=0).max() > (0.28 if case == "hard" else 0.12):
            return stream
    raise RuntimeError(f"Could not find an illustrative {case} stream.")


def wrong_label_rates(streams: np.ndarray) -> np.ndarray:
    """Return the empirical wrong-label rate after each sample count."""
    sample_counts = np.arange(1, streams.shape[1] + 1)
    running_means = np.cumsum(streams, axis=1) / sample_counts[None, :, None]
    return np.mean(np.argmax(running_means, axis=2) != 0, axis=0)


def simplex_coordinates(probabilities: np.ndarray) -> np.ndarray:
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)))
    return probabilities @ vertices


def draw_simplex(ax: plt.Axes, case: str, samples: np.ndarray) -> None:
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)))
    ax.plot(*np.vstack((vertices, vertices[0])).T, color="#333333", linewidth=2.2)
    coordinates = simplex_coordinates(samples)
    if case == "hard":
        first = samples[:, 0] > samples[:, 1]
        # Use one color so the two modes are read as one ambiguous posterior;
        # their separation is conveyed by their locations on the simplex.
        for mask in (first, ~first):
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=9, color=CLASS_COLORS[0], alpha=0.18, linewidths=0)
    else:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=9, color=CLASS_COLORS[0], alpha=0.18, linewidths=0)
    mean_xy = simplex_coordinates(true_mean(case)[None, :])[0]
    ax.scatter(*mean_xy, marker="*", s=70, color="#111111", zorder=4)
    for label, position, ha, va in (("class 1", vertices[0], "right", "top"), ("class 2", vertices[1], "left", "top"), ("class 3", vertices[2], "center", "bottom")):
        ax.text(*position, label, ha=ha, va=va, fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.09, 0.96)
    ax.axis("off")


def make_running_panel(
    ax: plt.Axes, case: str, stream: np.ndarray, *, show_ylabel: bool = True
) -> None:
    running = np.cumsum(stream, axis=0) / np.arange(1, len(stream) + 1)[:, None]
    x = np.arange(1, len(stream) + 1)
    reference = true_mean(case)
    for class_index, color in enumerate(CLASS_COLORS):
        ax.plot(x, running[:, class_index], color=color, linewidth=PAPER_LINE_WIDTH)
        ax.axhline(reference[class_index], color=color, linestyle=":", linewidth=1.8, alpha=0.9)
    ax.axvline(FIXED_SAMPLES, color="black", linestyle="--", linewidth=1.8)
    ax.text(FIXED_SAMPLES + 1.3, 0.96, r"$t=30$", fontsize=9.5, va="top", color="black")
    title = "Easy input" if case == "easy" else "Ambiguous input"
    ax.set_title(title, loc="center")
    ax.set_xlabel("Posterior samples")
    ax.set_ylabel("Class probability" if show_ylabel else "")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, len(stream))
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    ax.grid(False)


def make_error_panel(
    ax: plt.Axes, easy_streams: np.ndarray, hard_streams: np.ndarray
) -> None:
    sample_counts = np.arange(1, ERROR_HORIZON + 1)
    easy_rates = 100.0 * wrong_label_rates(easy_streams)
    hard_rates = 100.0 * wrong_label_rates(hard_streams)
    easy_color = CLASS_COLORS[0]
    hard_color = CLASS_COLORS[1]
    marker_samples = np.asarray((1, 5, 10, 20, 30, 50, 75, 100))

    ax.plot(sample_counts, easy_rates, color=easy_color, linewidth=2.4, label="easy input")
    ax.plot(sample_counts, hard_rates, color=hard_color, linewidth=2.4, label="ambiguous input")
    ax.scatter(marker_samples, easy_rates[marker_samples - 1], s=18, color=easy_color, zorder=3)
    ax.scatter(marker_samples, hard_rates[marker_samples - 1], s=18, color=hard_color, zorder=3)
    ax.axvline(FIXED_SAMPLES, color="black", linestyle="--", linewidth=1.8, zorder=1)

    easy_at_budget = easy_rates[FIXED_SAMPLES - 1]
    hard_at_budget = hard_rates[FIXED_SAMPLES - 1]
    zero_error_times = np.flatnonzero(easy_rates <= 0.0)
    first_zero_error = int(zero_error_times[0] + 1) if zero_error_times.size else ERROR_HORIZON
    ax.scatter(
        (FIXED_SAMPLES, FIXED_SAMPLES),
        (easy_at_budget, hard_at_budget),
        s=55,
        facecolor="white",
        edgecolor=(easy_color, hard_color),
        linewidth=2.0,
        zorder=4,
    )
    ax.annotate(
        f"{hard_at_budget:.1f}% still wrong",
        xy=(FIXED_SAMPLES, hard_at_budget),
        xytext=(45, 20),
        textcoords="data",
        ha="left",
        va="center",
        color=hard_color,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=hard_color, linewidth=1.4),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFF4F1", edgecolor=hard_color, linewidth=0.8),
    )
    ax.annotate(
        f"0% wrong by $t = {first_zero_error}$\n$\\Rightarrow\\ t = 30$ is unnecessary",
        xy=(FIXED_SAMPLES, easy_at_budget),
        xytext=(68, 6.0),
        textcoords="data",
        ha="center",
        va="center",
        color=easy_color,
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=easy_color, linewidth=1.4),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F1F7FC", edgecolor=easy_color, linewidth=0.8),
    )
    ax.text(FIXED_SAMPLES + 1.2, 39.0, r"$t=30$", ha="left", va="top", fontsize=9.5)

    ax.set_title("Probability of selecting the wrong label\n"
                "vs. number of posterior samples"
                 )
    ax.set_xlabel(r"Posterior samples $t$")
    ax.set_ylabel("Error rate (%)")
    ax.set_xlim(1 - 0.05 * (ERROR_HORIZON - 1), ERROR_HORIZON)
    # Match the small lower-axis clearance used by the allocation-density
    # panels, so the zero-valued curve remains visually distinct from the
    # bottom spine.
    ax.set_ylim(-2.0, 40.0)
    ax.set_xticks((1, 10, 30, 50, 75, 100))
    ax.set_yticks((0, 10, 20, 30, 40))
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.grid(False)


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(SCRIPT_DIR / f"{name}.pdf", transparent=True, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(SCRIPT_DIR / f"{name}.png", dpi=240, transparent=True)
    plt.close(figure)


def make_intro_figure() -> None:
    easy_streams = generate_streams("easy", N_RUNS, ERROR_HORIZON, SEED + 11)
    hard_streams = generate_streams("hard", N_RUNS, ERROR_HORIZON, SEED + 12)
    fig, ax = plt.subplots(figsize=(4.8, 3.45))
    make_error_panel(ax, easy_streams, hard_streams)
    fig.tight_layout()
    save_figure(fig, "intro_fixed_budget")


def make_intro_details_figure() -> None:
    """Combine the trajectory and simplex diagnostics for the appendix."""
    easy_stream = choose_illustrative_stream("easy", RUNNING_HORIZON)
    hard_stream = choose_illustrative_stream("hard", RUNNING_HORIZON)
    rng = np.random.default_rng(SEED + 21)
    easy_samples = sample_batch("easy", 1_000, rng)
    hard_samples = sample_batch("hard", 1_000, rng)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.4, 3.8),
        gridspec_kw={"wspace": 0.34, "width_ratios": (1.25, 1.25, 1.0, 1.0)},
    )
    make_running_panel(axes[0], "easy", easy_stream, show_ylabel=True)
    make_running_panel(axes[1], "hard", hard_stream, show_ylabel=False)
    axes[0].set_title("(a) Easy running mean", fontsize=10)
    axes[1].set_title("(b) Ambiguous running mean", fontsize=10)

    draw_simplex(axes[2], "easy", easy_samples)
    draw_simplex(axes[3], "hard", hard_samples)
    axes[2].set_title("(c) Easy simplex", fontsize=10, pad=5)
    axes[3].set_title("(d) Ambiguous simplex", fontsize=10, pad=5)

    handles = [
        Line2D([0], [0], color=CLASS_COLORS[0], lw=PAPER_LINE_WIDTH, label=r"running $\widehat\mu_1$"),
        Line2D([0], [0], color=CLASS_COLORS[1], lw=PAPER_LINE_WIDTH, label=r"running $\widehat\mu_2$"),
        Line2D([0], [0], color=CLASS_COLORS[2], lw=PAPER_LINE_WIDTH, label=r"running $\widehat\mu_3$"),
        Line2D([0], [0], color="#444444", lw=1.8, linestyle=":", label="true mean"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", label=r"$t=30$ budget"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.5,
        columnspacing=0.9,
        fontsize=9.5,
    )
    fig.subplots_adjust(bottom=0.29, left=0.05, right=0.975, top=0.90)
    save_figure(fig, "intro_fixed_budget_details")


def make_simplex_figure() -> None:
    rng = np.random.default_rng(SEED + 21)
    easy = sample_batch("easy", 1_000, rng)
    hard = sample_batch("hard", 1_000, rng)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), gridspec_kw={"wspace": 0.18})
    draw_simplex(axes[0], "easy", easy)
    draw_simplex(axes[1], "hard", hard)
    axes[0].set_title(r"(a) Easy: $\mathrm{Dirichlet}(5\,\mu)$", fontsize=10, pad=5)
    axes[1].set_title("(b) Hard: two-mode posterior disagreement", fontsize=10, pad=5)
    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=8, label="true mean"),
        Line2D([0], [0], marker="o", color=CLASS_COLORS[0], alpha=0.55, linestyle="none", markersize=4, label="posterior draws"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.16, left=0.02, right=0.98, top=0.88)
    save_figure(fig, "intro_simplex_distributions")


def main() -> None:
    configure_matplotlib()
    make_intro_figure()
    make_intro_details_figure()
    hard_streams = generate_streams("hard", N_RUNS, ERROR_HORIZON, SEED + 12)
    hard_at_budget = 100.0 * wrong_label_rates(hard_streams)[FIXED_SAMPLES - 1]
    (SCRIPT_DIR / "metrics.tex").write_text(
        "% Generated by generate.py; do not edit.\n"
        f"\\providecommand{{\\IntroHardWrongRatePct}}{{{hard_at_budget:.1f}}}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
