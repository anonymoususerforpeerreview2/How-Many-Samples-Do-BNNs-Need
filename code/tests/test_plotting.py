"""Regression checks for outcome-aware BNN figures."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnn_demo_lib.plotting import plot_stopping_histogram  # noqa: E402


class _CaptureWriter:
    def __init__(self) -> None:
        self.figure = None
        self.name = None

    def save(self, figure, name: str) -> None:
        self.figure = figure
        self.name = name


def test_stopping_histogram_stacks_all_three_outcomes() -> None:
    writer = _CaptureWriter()
    stop_times = np.array([2, 2, 2, 4, 4, 6])
    statuses = np.array(
        ["stopped", "abstained", "timeout", "stopped", "abstained", "timeout"]
    )

    plot_stopping_histogram(stop_times, statuses, 3, writer)

    assert writer.name == "poster_stop_hist"
    assert writer.figure is not None
    axis = writer.figure.axes[0]
    assert len(axis.patches) == 9
    assert sum(patch.get_height() for patch in axis.patches) == len(stop_times)
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Resolved",
        "Early abstention (provably unresolved)",
        r"Unresolved at $T_{\max}$",
    ]
    plt.close(writer.figure)
