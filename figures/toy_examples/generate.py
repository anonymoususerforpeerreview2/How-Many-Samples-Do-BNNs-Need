"""Generate the paper's easy/ambiguous toy-sequence figure.

Run ``python generate.py`` from this directory. Outputs are written beside this
script and consumed directly by ``main.tex``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


GRAPH_DIR = Path(__file__).resolve().parent
ROOT = GRAPH_DIR.parents[1]
sys.path.insert(0, str(ROOT / "code"))

import adaptive_cs as acs  # noqa: E402
from bnn_demo_lib.plotting import FigureWriter, configure_matplotlib, plot_toy_sequence_overview  # noqa: E402
from bnn_demo_lib.data import sample_predictive_around  # noqa: E402
from bnn_demo_lib.settings import config_from_omegaconf  # noqa: E402
from experiments.toy import (  # noqa: E402
    load_config,
    make_toy_data,
    paper_example_indices,
    paper_example_stopping_times,
    write_tex_macros,
)


def main() -> None:
    configure_matplotlib()
    cfg = load_config()
    data = make_toy_data(cfg)
    examples = paper_example_indices(data)
    hard, _ = examples
    # Keep the illustrative class-2/class-3 mixture inside the two-peak
    # setting's configured ranges, while showing its two distinct modes.
    hard_mean = np.array([0.10, 0.55, 0.35])
    hard_shift = 0.20
    hard_components = np.array([
        hard_mean + [0.0, hard_shift, -hard_shift],
        hard_mean - [0.0, hard_shift, -hard_shift],
    ]).round(12)
    hard_weights = np.array([0.5, 0.5])
    hard_concentration = 100.0
    two_peak = next(cohort for cohort in cfg.dataset.cohorts if cohort.kind == "two_peak")
    if not (
        float(two_peak.top_probability[0]) <= hard_mean[1] <= float(two_peak.top_probability[1])
        and float(two_peak.tail_probability[0]) <= hard_mean[0] <= float(two_peak.tail_probability[1])
        and float(two_peak.mode_shift[0]) <= hard_shift <= float(two_peak.mode_shift[1])
        and float(two_peak.kappa[0]) <= hard_concentration <= float(two_peak.kappa[1])
    ):
        raise ValueError("Illustrative mixture falls outside the simulated setting")
    rng = np.random.default_rng(321)
    hard_samples = np.asarray([
        sample_predictive_around(hard_components, hard_weights, hard_concentration, rng)
        for _ in range(int(cfg.experiment.max_samples))
    ])
    means = data.means.copy()
    concentrations = data.concentrations.copy()
    component_means = data.component_means.copy()
    component_weights = data.component_weights.copy()
    samples = data.samples.copy()
    means[hard] = hard_weights @ hard_components
    concentrations[hard] = hard_concentration
    component_means[hard] = hard_components
    component_weights[hard] = hard_weights
    samples[:, hard] = hard_samples
    figure_data = replace(
        data,
        means=means,
        concentrations=concentrations,
        component_means=component_means,
        component_weights=component_weights,
        samples=samples,
    )
    stopping = paper_example_stopping_times(figure_data, cfg, example_indices=examples)
    writer = FigureWriter(GRAPH_DIR, ("pdf",), 200)
    settings = config_from_omegaconf(cfg)
    plot_toy_sequence_overview(
        figure_data.means,
        figure_data.concentrations,
        figure_data.is_hard,
        lambda: acs.ClassSelection(config=settings, stop_on_resolve=False),
        int(cfg.experiment.seed),
        int(cfg.experiment.predictive_samples),
        int(cfg.experiment.max_samples),
        writer,
        stopping_times=stopping,
        component_means=figure_data.component_means,
        component_weights=figure_data.component_weights,
        samples=figure_data.samples,
        example_indices=examples,
        distribution_epsilon=float(cfg.experiment.distribution_epsilon),
        probability_threshold=float(cfg.experiment.probability_threshold),
    )
    source = GRAPH_DIR / "paper_toy_sequences.pdf"
    target = GRAPH_DIR / "toy_examples.pdf"
    source.replace(target)
    summary = {
        "seed": int(cfg.experiment.seed),
        "hard_mean": hard_mean.tolist(),
        "hard_components": hard_components.tolist(),
        "hard_concentration": hard_concentration,
        "stopping_times": stopping,
    }
    (GRAPH_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_tex_macros(
        GRAPH_DIR / "metrics.tex",
        {
            "ToyEasyClassStop": str(stopping["easy"]["class"]),
            "ToyEasyDistributionStop": str(stopping["easy"]["distribution"]),
            "ToyEasyThresholdStop": str(stopping["easy"]["threshold"]),
            "ToyHardDistributionStop": str(stopping["hard"]["distribution"]),
            "ToyHardThresholdStop": str(stopping["hard"]["threshold"]),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
