"""Evaluate safe early abstention on the paper's simulated toy dataset.

Run this argument-free script from this directory (or from anywhere); it writes the PDF and
JSON summary next to the script so that ``main.tex`` can load the figure with
the relative path ``graphs/early_stopping_evaluation/early_stopping_evaluation``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRAPH_DIR = Path(__file__).resolve().parent
REPO_DIR = GRAPH_DIR.parents[1]
CODE_DIR = REPO_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

import adaptive_cs as acs  # noqa: E402
from bnn_demo_lib.plotting import configure_matplotlib  # noqa: E402
from bnn_demo_lib.settings import config_from_omegaconf  # noqa: E402
from experiments.toy import load_config, make_toy_data  # noqa: E402


def _run_class_selection(samples: np.ndarray, cfg, early_abstention: bool):
    """Run the native tracker on a fixed stream, returning final arrays."""
    tracker = acs.ClassSelection(
        max_samples=samples.shape[0],
        config=config_from_omegaconf(cfg),
        early_abstention=early_abstention,
    )
    result = None
    while not tracker.done:
        active = tracker.active_indices
        if isinstance(active, slice):
            result = tracker.update(samples[tracker.time])
        else:
            result = tracker.update(samples[tracker.time, active], indices=active)
    assert result is not None
    return (
        result.n_samples.cpu().numpy(),
        result.prediction.cpu().numpy(),
        np.asarray(result.status),
    )


def main() -> None:
    configure_matplotlib()
    # Figure 7 is scaled from 4.8 in into .28\textwidth, while each Figure 8
    # plot is scaled from 3.4 in into .42 * .69\textwidth.
    panel_scale = (.42 * .69 / 3.4) / (.28 / 4.8)
    plt.rcParams.update({
        "font.size": 15 * panel_scale,
        "axes.labelsize": 16 * panel_scale,
        "xtick.labelsize": 13 * panel_scale,
        "ytick.labelsize": 13 * panel_scale,
        "axes.linewidth": 2.2 * panel_scale,
        "xtick.major.width": 2.2 * panel_scale,
        "ytick.major.width": 2.2 * panel_scale,
    })
    cfg = load_config()
    seed = int(cfg.experiment.seed)
    max_samples = int(cfg.experiment.max_samples)
    data = make_toy_data(cfg)
    means, labels, samples = data.means, data.labels, data.samples

    baseline_times, baseline_predictions, baseline_statuses = _run_class_selection(
        samples, cfg, early_abstention=False
    )
    early_times, early_predictions, early_statuses = _run_class_selection(
        samples, cfg, early_abstention=True
    )

    inconclusive = baseline_statuses != "stopped"
    savings = max_samples - early_times[inconclusive]
    early_resolved = early_statuses[inconclusive] == "abstained"
    # This is a diagnostic, so also report the capped predictions' conditional
    # accuracy without using it as a certified accuracy claim.
    summary = {
        "seed": seed,
        "num_inputs": int(len(means)),
        "baseline_inconclusive_count": int(np.sum(inconclusive)),
        "early_abstained_count": int(np.sum(early_resolved)),
        "early_still_inconclusive_count": int(np.sum(~early_resolved)),
        "mean_samples_saved": float(np.mean(savings)) if len(savings) else 0.0,
        "median_samples_saved": float(np.median(savings)) if len(savings) else 0.0,
        "q25_samples_saved": float(np.quantile(savings, 0.25)) if len(savings) else 0.0,
        "q75_samples_saved": float(np.quantile(savings, 0.75)) if len(savings) else 0.0,
        "max_samples_saved": int(np.max(savings)) if len(savings) else 0,
        "baseline_certified_accuracy": float(np.mean(
            baseline_predictions[baseline_statuses == "stopped"]
            == labels[baseline_statuses == "stopped"]
        )),
        "early_certified_accuracy": float(np.mean(
            early_predictions[early_statuses == "stopped"]
            == labels[early_statuses == "stopped"]
        )),
    }

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    plot_max = max(20, int(np.ceil(np.max(savings) / 10.0) * 10))
    bin_width = 1
    bins = np.arange(-0.5, plot_max + bin_width + 0.5, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    weights = np.full(savings.shape, 1.0 / len(savings), dtype=float)
    # ax.hist(savings, bins=bins, color="#4575b4", edgecolor="white")
    ax.hist(
        savings, bins=bins, weights=weights, color="black", edgecolor="white"
    )
    ax.axvline(
        np.mean(savings), color="#d73027", linestyle="--",
        linewidth=1.4 * panel_scale,
        label=f"Mean = {np.mean(savings):.1f}",
    )
    # ax.axvline(
    #     np.median(savings), color="#F8AC3A", linestyle=":", linewidth=1.4,
    #     label=f"Median = {np.median(savings):.0f}",
    # )
    ax.set(
        xlabel="Samples saved",
        ylabel="Ratio",
        xlim=(-1, plot_max + 1),
    )
    ax.set_xticks(bin_centers[::2])
    ax.set_title("Early abstention savings", fontsize=17.5 * panel_scale)
    ax.legend(frameon=False, fontsize=9.5 * panel_scale)
    fig.subplots_adjust(left=.195, right=.99, bottom=.205, top=.87)
    fig.savefig(GRAPH_DIR / "early_stopping_evaluation.pdf", dpi=200)
    plt.close(fig)
    (GRAPH_DIR / "early_stopping_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (GRAPH_DIR / "metrics.tex").write_text(
        "% Generated by early_stopping_evaluation.py; do not edit.\n"
        f"\\providecommand{{\\EarlyInconclusiveCount}}{{{summary['baseline_inconclusive_count']}}}\n"
        f"\\providecommand{{\\EarlyAbstainedCount}}{{{summary['early_abstained_count']}}}\n"
        f"\\providecommand{{\\EarlyStillCappedCount}}{{{summary['early_still_inconclusive_count']}}}\n"
        f"\\providecommand{{\\EarlyMeanSaved}}{{{summary['mean_samples_saved']:.1f}}}\n"
        f"\\providecommand{{\\EarlyMedianSaved}}{{{summary['median_samples_saved']:.0f}}}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
