"""Generate the illustrative future-function bound used in Section 5.

The parameters immediately below are intentionally kept in one place so that
the example can be adjusted without touching the plotting code.  The figure is
illustrative: ``WEALTH_CUTOFF`` is a fixed proxy for the time-varying
coordinate threshold gamma_t^c in Equation (8).
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


GRAPH_DIR = Path(__file__).resolve().parent
REPO_DIR = GRAPH_DIR.parents[1]
CODE_DIR = REPO_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

from non_vectorized_class_selection import (  # noqa: E402
    ConfidenceSequenceSettings,
    clipped_bets,
    scalar_wealth_path,
    unconstrained_bet,
)


# ---------------------------------------------------------------------------
# Editable example settings
# ---------------------------------------------------------------------------
CURRENT_TIME = 50
MAX_TIME = 100
CANDIDATE_MEAN = 0.50
ALPHA = 0.05
CLIP = 0.08
HEDGE_WEIGHT = 0.50
WEALTH_CUTOFF = 150.0
SEED = 20260916
WEALTH_SETTINGS = ConfidenceSequenceSettings(
    alpha=ALPHA,
    c=CLIP,
    theta=HEDGE_WEIGHT,
)

# The prefix is concentrated slightly above the candidate mean, so its wealth
# grows steadily while remaining below the illustrative elimination cutoff.
OBSERVED_STREAM = ("observed", 0.68, 500.0)

# Each tuple is (label, kind, first parameter, second parameter).  ``beta``
# streams use (mean, concentration); ``switch`` uses the means before/after
# the midpoint and a fixed concentration.  Add or edit rows to change the
# possible future sampling paths in the plot.
FUTURE_STREAMS = (
    ("strong upward evidence", "beta", 0.90, 120.0),
    ("continued mild evidence", "beta", 0.68, 500.0),
    ("noisy, centered at m", "beta", 0.50, 35.0),
    ("early dip, then rebound", "switch", 0.38, 0.68),
    ("late downward shift", "beta", 0.40, 100.0),
)

OUTPUT_STEM = "early_stopping_bound"


def configure_matplotlib() -> None:
    """Match the typography and line weights of the other paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 9,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 2.2,
            "xtick.major.width": 2.2,
            "ytick.major.width": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def beta_stream(
    mean: float, concentration: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a reproducible sequence in [0, 1] with the requested mean."""
    return rng.beta(mean * concentration, (1.0 - mean) * concentration, size)


def optimistic_future_bound(prefix: np.ndarray) -> np.ndarray:
    """Evaluate the future-function upper bound from Section 5.2."""
    prefix_result = scalar_wealth_path(
        prefix,
        CANDIDATE_MEAN,
        WEALTH_SETTINGS,
    )
    plus = float(prefix_result.positive[-1])
    minus = float(prefix_result.negative[-1])
    variance_numerator = prefix_result.variance_numerator

    bound = [float(prefix_result.wealth[-1])]
    for future_time in range(CURRENT_TIME + 1, MAX_TIME + 1):
        upper_stake = float(
            unconstrained_bet(ALPHA, variance_numerator, future_time)
        )
        plus_stake = float(
            clipped_bets(upper_stake, CANDIDATE_MEAN, CLIP, component=0)
        )
        minus_stake = float(
            clipped_bets(upper_stake, CANDIDATE_MEAN, CLIP, component=1)
        )

        # The positive and negative processes are bounded separately by their
        # own most favorable observation, y=1 and y=0, respectively.
        plus *= 1.0 + plus_stake * (1.0 - CANDIDATE_MEAN)
        minus *= 1.0 + minus_stake * CANDIDATE_MEAN
        bound.append(max(HEDGE_WEIGHT * plus, (1.0 - HEDGE_WEIGHT) * minus))

    return np.asarray(bound)


def make_future_streams(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Generate varied but reproducible counterfactual continuations."""
    length = MAX_TIME - CURRENT_TIME
    streams: list[tuple[str, np.ndarray]] = []
    for label, kind, first, second in FUTURE_STREAMS:
        if kind == "beta":
            future = beta_stream(first, second, length, rng)
        elif kind == "switch":
            midpoint = length // 2
            future = np.concatenate(
                (
                    beta_stream(first, 140.0, midpoint, rng),
                    beta_stream(second, 140.0, length - midpoint, rng),
                )
            )
        else:
            raise ValueError(f"Unknown future-stream kind: {kind}")
        streams.append((label, future))
    return streams


def save_wealth_figure(
    prefix: np.ndarray,
    bound: np.ndarray,
    future_streams: list[tuple[str, np.ndarray]],
    *,
    output_stem: str,
    future_legend: str,
    future_region_label: str,
    distinguish_future_styles: bool,
) -> None:
    """Plot one observed prefix, possible continuations, and their bound."""
    prefix_wealth = scalar_wealth_path(
        prefix,
        CANDIDATE_MEAN,
        WEALTH_SETTINGS,
    ).wealth

    observed_times = np.arange(1, CURRENT_TIME + 1)
    future_times = np.arange(CURRENT_TIME, MAX_TIME + 1)

    figure, axis = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)
    axis.axvspan(CURRENT_TIME, MAX_TIME, color="#E8E8E8", alpha=0.55, zorder=0)

    axis.plot(
        observed_times,
        prefix_wealth,
        # color="#2878B5",
        color="black",
        linewidth=2.1,
        zorder=4,
    )
    for path_index, (_, future) in enumerate(future_streams):
        full_wealth = scalar_wealth_path(
            np.concatenate((prefix, future)),
            CANDIDATE_MEAN,
            WEALTH_SETTINGS,
        ).wealth
        path_values = np.concatenate(
            ([prefix_wealth[-1]], full_wealth[CURRENT_TIME:])
        )
        if np.any(path_values > bound * (1.0 + 1e-12)):
            raise RuntimeError("A simulated path exceeded the optimistic bound.")
        axis.plot(
            future_times,
            path_values,
            color="#75828B",
            alpha=0.58 if distinguish_future_styles else 0.30,
            linewidth=0.95,
            linestyle=(0, (2 + path_index, 1.5))
            if distinguish_future_styles
            else "-",
            label=future_legend if path_index == 0 else None,
            zorder=2,
        )

    axis.plot(
        future_times,
        bound,
        color="#C82423",
        linewidth=2.1,
        zorder=5,
    )
    axis.axhline(
        WEALTH_CUTOFF,
        color="black",
        linewidth=1.1,
        linestyle=(0, (5, 3)),
        zorder=1,
    )
    axis.axvline(CURRENT_TIME, color="0.25", linewidth=0.9, linestyle=":", zorder=1)
    axis.text(
        CURRENT_TIME - 1.5,
        0.035,
        rf"$t={CURRENT_TIME}$",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        color="0.25",
        fontsize=7,
    )
    axis.text(
        (CURRENT_TIME + MAX_TIME) / 2,
        0.965,
        future_region_label,
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        color="0.35",
        fontsize=7,
    )
    axis.text(
        CURRENT_TIME - 1.5,
        prefix_wealth[-1] * 1.08,
        r"observed $k_t(m_0)$",
        # color="#2878B5",
        color="black",
        ha="right",
        va="bottom",
        fontsize=6.8,
    )
    bound_label_time = MAX_TIME - 8
    bound_label_index = bound_label_time - CURRENT_TIME
    axis.text(
        bound_label_time,
        bound[bound_label_index] * 1.08,
        r"optimistic $\overline{k}_t(m_0)$",
        color="#C82423",
        ha="right",
        va="bottom",
        fontsize=6.8,
    )
    axis.text(
        MAX_TIME - 1,
        WEALTH_CUTOFF * 1.05,
        r"$\gamma^c$",
        color="black",
        ha="right",
        va="bottom",
        fontsize=7,
    )

    axis.set_yscale("log")
    axis.set_xlim(1, MAX_TIME)
    axis.set_ylim(0.35, max(WEALTH_CUTOFF * 2.4, bound.max() * 1.12))
    axis.set_xticks((25, CURRENT_TIME, 75, MAX_TIME))
    axis.set_xlabel(r"Time $t$ (number of observations)")
    axis.set_ylabel(r"Function value of $k_t$ for a fixed $m_0$")
    # axis.set_ylabel(r"Function value for fixed $m$, $k_t(m_0)$")
    axis.grid(axis="y", which="major", color="0.88", linewidth=0.55)
    axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
        borderpad=0.2,
        ncol=1,
    )

    output_dir = Path(__file__).resolve().parent
    common_options = {"bbox_inches": "tight", "transparent": True}
    figure.savefig(
        output_dir / f"{output_stem}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
        **common_options,
    )
    figure.savefig(output_dir / f"{output_stem}.png", dpi=220, **common_options)
    plt.close(figure)


def main() -> None:
    if not 0.0 < CANDIDATE_MEAN < 1.0:
        raise ValueError("CANDIDATE_MEAN must lie strictly between zero and one.")
    if not 1 <= CURRENT_TIME < MAX_TIME:
        raise ValueError("Require 1 <= CURRENT_TIME < MAX_TIME.")
    configure_matplotlib()
    rng = np.random.default_rng(SEED)
    _, observed_mean, observed_concentration = OBSERVED_STREAM
    prefix = beta_stream(observed_mean, observed_concentration, CURRENT_TIME, rng)
    bound = optimistic_future_bound(prefix)

    save_wealth_figure(
        prefix,
        bound,
        make_future_streams(rng),
        output_stem=OUTPUT_STEM,
        future_legend=r"possible future $k_t(m_0)$",
        future_region_label="unseen observations",
        distinguish_future_styles=True,
    )


if __name__ == "__main__":
    main()
