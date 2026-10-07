"""Matplotlib styling shared by paper figures without model dependencies."""

import matplotlib.pyplot as plt


COLORS = ("#2878B5", "#C82423", "#F8AC3A")
PAPER_LINE_WIDTH = 2.2
PAPER_GUIDE_LINE_WIDTH = 0.8


def configure_matplotlib() -> None:
    """Apply a consistent paper-friendly style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": PAPER_LINE_WIDTH,
            "lines.linewidth": PAPER_LINE_WIDTH,
            "lines.markersize": 6,
            "xtick.major.width": PAPER_LINE_WIDTH,
            "ytick.major.width": PAPER_LINE_WIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
