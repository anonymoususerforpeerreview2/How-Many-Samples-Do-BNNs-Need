"""Plot one generated input from each of the four simulated settings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


GRAPH_DIR = Path(__file__).resolve().parent
ROOT = GRAPH_DIR.parents[1]
sys.path.insert(0, str(ROOT / "code"))

from bnn_demo_lib.plotting import configure_matplotlib  # noqa: E402
from experiments.toy import load_config, make_toy_data, write_tex_macros  # noqa: E402


SETTINGS = (
    ("confident_stable", "(a) Confident", "Confident", "#2878B5"),
    ("threshold_boundary", "(b) Threshold", "Threshold", "#5B3F99"),
    ("diffuse_near_tie", "(c) Near tie", "NearTie", "#C82423"),
    ("multimodal_disagreement", "(d) Two modes", "TwoMode", "#D08000"),
)
VERTICES = np.array(((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)))


def _select_example(data, cohort) -> int:
    """Select a central input with class 1 leading and class 2 second."""
    means = data.means
    candidates = np.flatnonzero(
        (data.cohorts == str(cohort.name))
        & (means.argmax(axis=1) == 0)
        & (np.argsort(means, axis=1)[:, -2] == 1)
    )
    if not len(candidates):
        raise ValueError(f"No suitable example for {cohort.name}")
    top_range = np.asarray(cohort.top_probability, dtype=float)
    kappa_range = np.asarray(cohort.kappa, dtype=float)
    # A lower leading probability makes the two favored classes visibly swap
    # between the modes in the final panel.
    top_target = (
        top_range[0] + 0.15 * np.ptp(top_range)
        if str(cohort.kind) == "two_peak" else top_range.mean()
    )
    score = (
        ((means[candidates, 0] - top_target) / np.ptp(top_range)) ** 2
        + ((data.concentrations[candidates] - kappa_range.mean()) / np.ptp(kappa_range)) ** 2
    )
    if str(cohort.kind) == "near_tie":
        margin_range = np.asarray(cohort.leader_margin, dtype=float)
        margins = means[candidates, 0] - means[candidates, 1]
        score += ((margins - margin_range.mean()) / np.ptp(margin_range)) ** 2
    elif str(cohort.kind) == "two_peak":
        tail_range = np.asarray(cohort.tail_probability, dtype=float)
        shift_range = np.asarray(cohort.mode_shift, dtype=float)
        shifts = data.component_means[candidates, 0, 0] - means[candidates, 0]
        score += ((means[candidates, 2] - tail_range.mean()) / np.ptp(tail_range)) ** 2
        score += ((shifts - shift_range.mean()) / np.ptp(shift_range)) ** 2
    return int(candidates[np.argmin(score)])


def _draw_panel(axis, samples: np.ndarray, mean: np.ndarray, components: np.ndarray,
                color: str, title: str, *, two_mode: bool) -> None:
    closed = np.vstack((VERTICES, VERTICES[0]))
    axis.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.8)
    points = samples @ VERTICES
    axis.scatter(points[:, 0], points[:, 1], s=15, alpha=0.25, color=color,
                 edgecolors="none", rasterized=True)
    axis.scatter(*(mean @ VERTICES), marker="*", s=200, color="black", zorder=5)
    if two_mode:
        locations = components @ VERTICES
        axis.scatter(locations[:, 0], locations[:, 1], marker="D", s=70,
                     facecolors="none", edgecolors="black", linewidths=1.5, zorder=4)
    for label, x, y, ha, va in (
        ("class 1", 0.04, -0.05, "left", "top"),
        ("class 2", 0.96, -0.05, "right", "top"),
        ("class 3", 0.50, 0.91, "center", "bottom"),
    ):
        axis.text(x, y, label, ha=ha, va=va, fontsize=14)
    axis.set(xlim=(-0.10, 1.10), ylim=(-0.12, 0.99))
    axis.set_title(title, fontsize=17, pad=8)
    axis.set_aspect("equal")
    axis.axis("off")


def _vector(values: np.ndarray) -> str:
    return "(" + ",".join(f"{value:.6f}" for value in values) + ")"


def main() -> None:
    configure_matplotlib()
    cfg = load_config()
    data = make_toy_data(cfg)
    cohort_by_name = {str(cohort.name): cohort for cohort in cfg.dataset.cohorts}
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.2))
    macros: dict[str, str] = {}
    summary: dict[str, object] = {"seed": int(cfg.experiment.seed), "examples": {}}
    for axis, (name, title, macro, color) in zip(axes, SETTINGS):
        index = _select_example(data, cohort_by_name[name])
        mean = data.means[index]
        components = data.component_means[index]
        concentration = float(data.concentrations[index])
        _draw_panel(axis, data.samples[:, index], mean, components, color, title,
                    two_mode=name == "multimodal_disagreement")
        macros[f"ToySetting{macro}Mean"] = _vector(mean)
        macros[f"ToySetting{macro}Kappa"] = f"{concentration:.6f}"
        if name == "multimodal_disagreement":
            macros["ToySettingTwoModeFirst"] = _vector(components[0])
            macros["ToySettingTwoModeSecond"] = _vector(components[1])
        summary["examples"][name] = {
            "index": index,
            "mean": mean.tolist(),
            "kappa": concentration,
            "component_means": components.tolist(),
            "component_weights": data.component_weights[index].tolist(),
            "num_draws": len(data.samples),
        }
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88,
                        wspace=0.14)
    fig.savefig(GRAPH_DIR / "toy_settings.pdf", bbox_inches="tight", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    write_tex_macros(GRAPH_DIR / "metrics.tex", macros)
    (GRAPH_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
