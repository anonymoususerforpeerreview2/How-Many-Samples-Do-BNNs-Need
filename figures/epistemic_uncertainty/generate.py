"""Generate the paper's epistemic/aleatoric uncertainty illustration."""

from __future__ import annotations

import json
import sys
from pathlib import Path


GRAPH_DIR = Path(__file__).resolve().parent
ROOT = GRAPH_DIR.parents[1]
sys.path.insert(0, str(ROOT / "code"))

from bnn_demo_lib.plotting import FigureWriter, configure_matplotlib, plot_epistemic_uncertainty  # noqa: E402
from bnn_demo_lib.settings import config_from_omegaconf  # noqa: E402
from experiments.toy import load_config, write_tex_macros  # noqa: E402


def main() -> None:
    configure_matplotlib()
    cfg = load_config()
    writer = FigureWriter(GRAPH_DIR, ("pdf",), 200)
    summary = plot_epistemic_uncertainty(
        config_from_omegaconf(cfg), int(cfg.experiment.seed), 1_000, writer
    )
    source = GRAPH_DIR / "paper_epistemic_uncertainty.pdf"
    source.replace(GRAPH_DIR / "epistemic_uncertainty.pdf")
    (GRAPH_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    stable = summary["stable_posterior"]
    disagreeing = summary["disagreeing_posterior"]
    write_tex_macros(
        GRAPH_DIR / "metrics.tex",
        {
            "StableEpistemicReference": f"{stable['reference_epistemic_nats']:.5f}",
            "DisagreeingEpistemicReference": f"{disagreeing['reference_epistemic_nats']:.5f}",
            "StableAleatoricReference": f"{stable['reference_aleatoric_nats']:.5f}",
            "DisagreeingAleatoricReference": f"{disagreeing['reference_aleatoric_nats']:.5f}",
            "SharedTotalReference": f"{stable['reference_total_nats']:.5f}",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
