"""Reproduce the synthetic TOMS 748 clipping-breakpoint analysis."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))

from bnn_demo_lib.plotting import configure_matplotlib  # noqa: E402
from adaptive_cs._cpp import CppRootTracer  # noqa: E402
from experiments.toy import load_config, make_toy_data  # noqa: E402


TYPE_LABELS = {
    "crossing": r"Minimizer $m^\star$",
    "lower": r"Lower endpoint $L_t$",
    "upper": r"Upper endpoint $U_t$",
}
TYPE_COLORS = {"crossing": "#3585C9", "lower": "#D55E00", "upper": "#009E73"}
TIME_COLORS = ["#440154", "#3B528B", "#21918C", "#5DC863", "#FDE725"]


def _unconstrained_bets(samples: np.ndarray, alpha: float) -> np.ndarray:
    """Match the production tracker's predictable stake update."""
    bets = np.empty_like(samples, dtype=float)
    sum_x = np.zeros(samples.shape[1:], dtype=float)
    sum_dev = np.zeros_like(sum_x)
    variance = np.full_like(sum_x, 0.25)
    numerator = 2.0 * math.log(2.0 / alpha)
    for index, value in enumerate(samples, start=1):
        denominator = variance * index * math.log1p(index)
        bets[index - 1] = np.sqrt(numerator / np.maximum(denominator, 1e-12))
        regularized_mean = (0.5 + sum_x + value) / (index + 1.0)
        sum_dev += np.square(value - regularized_mean)
        sum_x += value
        variance = (0.25 + sum_dev) / (index + 1.0)
    return bets


def _log_component(
    observations: np.ndarray,
    bets: np.ndarray,
    candidate: float,
    component: str,
    clip: float,
    theta: float,
) -> float:
    plus = component == "plus"
    denominator = candidate if plus else 1.0 - candidate
    cap = math.inf if denominator == 0.0 else clip / denominator
    value = math.log(theta if plus else 1.0 - theta)
    for observation, unconstrained in zip(observations, bets):
        stake = min(float(unconstrained), cap)
        centered = float(observation) - candidate
        value += math.log1p((stake if plus else -stake) * centered)
    return value


def _breakpoints(bets: np.ndarray, clip: float, call_type: str) -> np.ndarray:
    active = bets > clip
    plus = clip / bets[active]
    minus = 1.0 - plus
    if call_type == "lower":
        values = plus
    elif call_type == "upper":
        values = minus
    else:
        values = np.concatenate([plus, minus])
    values = values[(values > 0.0) & (values < 1.0)]
    return np.unique(values)


def _contains(lower: float, upper: float, breakpoints: np.ndarray) -> bool:
    if breakpoints.size == 0:
        return False
    index = int(np.searchsorted(breakpoints, lower, side="left"))
    return index < breakpoints.size and bool(breakpoints[index] <= upper)


def _safe_quantile(values: Iterable[float], probability: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, probability)) if array.size else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    clean = [row for row in rows if not row["censored"]]
    clean_updates = [row["first_clean_update"] for row in clean]
    eval_to_clean = [row["evaluations_to_clean"] for row in clean]
    fractions = [row["fraction_evaluations_to_clean"] for row in clean]
    total_evaluations = [row["total_evaluations"] for row in rows]
    return {
        "calls": n,
        "initially_clean_pct": 100.0
        * sum(row["first_clean_update"] == 0 for row in clean)
        / n,
        "became_clean_before_termination_pct": 100.0
        * sum(
            row["first_clean_update"] is not None
            and row["first_clean_update"] > 0
            for row in rows
        )
        / n,
        "terminated_before_clean_pct": 100.0 * sum(row["censored"] for row in rows) / n,
        "first_clean_update_conditional": {
            "q25": _safe_quantile(clean_updates, 0.25),
            "median": _safe_quantile(clean_updates, 0.50),
            "q75": _safe_quantile(clean_updates, 0.75),
            "q90": _safe_quantile(clean_updates, 0.90),
            "q95": _safe_quantile(clean_updates, 0.95),
            "q99": _safe_quantile(clean_updates, 0.99),
        },
        "evaluations_to_clean_conditional": {
            "q25": _safe_quantile(eval_to_clean, 0.25),
            "median": _safe_quantile(eval_to_clean, 0.50),
            "q75": _safe_quantile(eval_to_clean, 0.75),
            "q90": _safe_quantile(eval_to_clean, 0.90),
        },
        "total_evaluations": {
            "mean": float(np.mean(total_evaluations)),
            "q25": _safe_quantile(total_evaluations, 0.25),
            "median": _safe_quantile(total_evaluations, 0.50),
            "q75": _safe_quantile(total_evaluations, 0.75),
            "q90": _safe_quantile(total_evaluations, 0.90),
        },
        "fraction_evaluations_to_clean_conditional": {
            "mean": float(np.mean(fractions)) if fractions else None,
            "median": _safe_quantile(fractions, 0.50),
            "q90": _safe_quantile(fractions, 0.90),
        },
        "empty_breakpoint_calls": sum(row["breakpoint_count"] == 0 for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    by_time = summary["by_observation_count"]
    macros = {
        "TomsBreakpointCalls": f"{overall['calls']:,}",
        "TomsInitiallyCleanPct": f"{overall['initially_clean_pct']:.1f}",
        "TomsBecameCleanPct": f"{overall['became_clean_before_termination_pct']:.1f}",
        "TomsCensoredPct": f"{overall['terminated_before_clean_pct']:.1f}",
        "TomsCensoredCount": str(round(overall["calls"] * overall["terminated_before_clean_pct"] / 100.0)),
        "TomsMedianCleanUpdate": f"{overall['first_clean_update_conditional']['median']:.0f}",
        "TomsMedianEvaluations": f"{overall['total_evaluations']['median']:.0f}",
        "TomsMedianEvalFractionPct": f"{100.0 * overall['fraction_evaluations_to_clean_conditional']['median']:.1f}",
        "TomsEmptyBreakpointCalls": str(overall["empty_breakpoint_calls"]),
        "TomsInitialTwentyPct": f"{by_time['20']['initially_clean_pct']:.1f}",
        "TomsInitialFourHundredPct": f"{by_time['400']['initially_clean_pct']:.1f}",
        "TomsCensoredMinPct": f"{min(item['terminated_before_clean_pct'] for item in by_time.values()):.2f}",
        "TomsCensoredMaxPct": f"{max(item['terminated_before_clean_pct'] for item in by_time.values()):.2f}",
    }
    lines = ["% Generated by generate.py; do not edit manually."]
    lines.extend(f"\\providecommand{{\\{key}}}{{{value}}}" for key, value in macros.items())
    lines.extend(["", "% Numeric cells for summary_table.tex."])
    for key, item in (
        ("Overall", summary["overall"]),
        *((name.title(), summary["by_call_type"][name]) for name in TYPE_LABELS),
    ):
        table_values = {
            "Calls": f"{item['calls']:,}",
            "InitiallyCleanPct": f"{item['initially_clean_pct']:.1f}",
            "BecameCleanPct": f"{item['became_clean_before_termination_pct']:.1f}",
            "TerminatedBeforeCleanPct": f"{item['terminated_before_clean_pct']:.2f}",
            "MedianCleanUpdate": f"{item['first_clean_update_conditional']['median']:.0f}",
            "MedianEvaluations": f"{item['total_evaluations']['median']:.0f}",
        }
        lines.extend(
            f"\\providecommand{{\\TomsTable{key}{suffix}}}{{{value}}}"
            for suffix, value in table_values.items()
        )
    (HERE / "metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_survival(
    calls: list[dict[str, Any]],
    bracket_map: dict[int, list[dict[str, Any]]],
    observation_times: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.45), sharey=True)
    for axis, call_type in zip(axes, TYPE_LABELS):
        type_calls = [row for row in calls if row["call_type"] == call_type]
        max_iteration = max(len(bracket_map[row["call_id"]]) for row in type_calls) - 1
        groups: list[tuple[str, list[dict[str, Any]], str, float]] = [
            ("All $t$", type_calls, "black", 2.2)
        ]
        for time, color in zip(observation_times, TIME_COLORS):
            groups.append(
                (f"$t={time}$", [row for row in type_calls if row["observation_count"] == time], color, 1.25)
            )
        for label, group, color, linewidth in groups:
            x_values: list[int] = []
            y_values: list[float] = []
            for iteration in range(max_iteration + 1):
                numerator = 0
                at_risk = 0
                for call in group:
                    trace = bracket_map[call["call_id"]]
                    if iteration < len(trace):
                        point = trace[iteration]
                        at_risk += 1
                    else:
                        point = trace[-1]
                    numerator += int(point["contains_breakpoint"])
                fraction = numerator / len(group)
                x_values.append(iteration)
                y_values.append(fraction)
                rows.append(
                    {
                        "call_type": call_type,
                        "observation_count": "all" if label == "All $t$" else int(label[3:-1]),
                        "iteration": iteration,
                        "calls": len(group),
                        "at_risk": at_risk,
                        "terminal_bracket_carried": len(group) - at_risk,
                        "contains_breakpoint": numerator,
                        "fraction": fraction,
                    }
                )
            axis.plot(x_values, y_values, marker="o", ms=2.8, lw=linewidth, color=color, label=label)
        axis.set_title(TYPE_LABELS[call_type])
        axis.set_xlabel("TOMS 748 bracket update")
        axis.set_ylim(-0.025, 1.025)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction containing a breakpoint")
    axes[-1].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(
        HERE / "breakpoint_survival.pdf",
        dpi=250,
        transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return rows


def main() -> None:
    configure_matplotlib()
    toy_cfg = load_config()
    local_cfg = OmegaConf.load(HERE / "config.yaml")
    data = make_toy_data(toy_cfg)
    observation_times = [int(value) for value in local_cfg.observation_times]
    alpha = float(toy_cfg.confidence.alpha)
    clip = float(toy_cfg.confidence.c)
    theta = float(toy_cfg.confidence.theta)
    padding = float(toy_cfg.confidence.padding)
    production = local_cfg.production
    reference = local_cfg.reference
    bets_all = _unconstrained_bets(data.samples, alpha)
    tracer = CppRootTracer()

    calls: list[dict[str, Any]] = []
    brackets: list[dict[str, Any]] = []
    bracket_map: dict[int, list[dict[str, Any]]] = {}
    validation_failures: list[str] = []
    call_id = 0

    def trace_root(observations: np.ndarray, bets: np.ndarray, call_type: str, level: float, center: float, settings: Any) -> dict[str, Any]:
        return tracer.trace(
            observations,
            bets,
            call_type,
            level=level,
            center=center,
            c=clip,
            theta=theta,
            x_tolerance=float(settings.x_tolerance),
            y_tolerance=float(settings.y_tolerance),
            max_steps=int(settings.max_steps),
        )

    def add_call(
        path_index: int,
        class_index: int,
        time: int,
        call_type: str,
        level: float,
        center: float,
        production_trace: dict[str, Any],
        reference_trace: dict[str, Any],
        breakpoints: np.ndarray,
    ) -> None:
        nonlocal call_id
        current_id = call_id
        call_id += 1
        root = 0.5 * (reference_trace["root_lower"] + reference_trace["root_upper"])

        trace_rows: list[dict[str, Any]] = []
        for iteration, (lower, upper, evaluations) in enumerate(
            zip(
                production_trace["lower"],
                production_trace["upper"],
                production_trace["function_evaluations"],
            )
        ):
            trace_rows.append(
                {
                    "call_id": current_id,
                    "path_index": path_index,
                    "class_index": class_index,
                    "observation_count": time,
                    "call_type": call_type,
                    "iteration": iteration,
                    "lower": float(lower),
                    "upper": float(upper),
                    "width": float(upper - lower),
                    "function_evaluations": int(evaluations),
                    "contains_breakpoint": int(_contains(float(lower), float(upper), breakpoints)),
                }
            )
        clean_indices = [row["iteration"] for row in trace_rows if not row["contains_breakpoint"]]
        first_clean_update = min(clean_indices) if clean_indices else None
        censored = first_clean_update is None
        evaluations_to_clean = (
            None if censored else trace_rows[first_clean_update]["function_evaluations"]
        )
        total_evaluations = int(production_trace["total_function_evaluations"])

        lower_values = np.asarray(production_trace["lower"])
        upper_values = np.asarray(production_trace["upper"])
        if np.any(np.diff(lower_values) < -2e-15) or np.any(np.diff(upper_values) > 2e-15):
            validation_failures.append(f"non-nested bracket for call {current_id}")
        if not np.all((lower_values - 2e-14 <= root) & (root <= upper_values + 2e-14)):
            validation_failures.append(f"reference root left bracket for call {current_id}")
        if not (
            production_trace["root_lower"] - 2e-14 <= reference_trace["root_lower"]
            and reference_trace["root_upper"] <= production_trace["root_upper"] + 2e-14
        ):
            validation_failures.append(f"reference bracket not inside production bracket for call {current_id}")

        call_row = {
            "call_id": current_id,
            "path_index": path_index,
            "class_index": class_index,
            "cohort": str(data.cohorts[path_index]),
            "observation_count": time,
            "call_type": call_type,
            "level": level if math.isfinite(level) else None,
            "center": center,
            "breakpoint_count": int(breakpoints.size),
            "reference_root": root,
            "reference_lower": float(reference_trace["root_lower"]),
            "reference_upper": float(reference_trace["root_upper"]),
            "first_clean_update": first_clean_update,
            "censored": int(censored),
            "terminal_iteration": len(trace_rows) - 1,
            "evaluations_to_clean": evaluations_to_clean,
            "total_evaluations": total_evaluations,
            "fraction_evaluations_to_clean": None if censored else evaluations_to_clean / total_evaluations,
        }
        calls.append(call_row)
        brackets.extend(trace_rows)
        bracket_map[current_id] = trace_rows

    for time in observation_times:
        print(f"Tracing t={time}...", flush=True)
        for path_index in range(data.samples.shape[1]):
            crossing_information: list[tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], float, float]] = []
            minimum_lower_wealth: list[float] = []
            for class_index in range(data.samples.shape[2]):
                observations = np.ascontiguousarray(data.samples[:time, path_index, class_index])
                bets = np.ascontiguousarray(bets_all[:time, path_index, class_index])
                prod = trace_root(observations, bets, "crossing", 1.0, 0.5, production)
                ref = trace_root(observations, bets, "crossing", 1.0, 0.5, reference)
                left = float(prod["root_lower"])
                right = float(prod["root_upper"])
                lower_wealth = max(
                    math.exp(min(700.0, _log_component(observations, bets, left, "minus", clip, theta))),
                    math.exp(min(700.0, _log_component(observations, bets, right, "plus", clip, theta))),
                )
                center = 0.5 * (left + right)
                crossing_information.append((observations, bets, prod, ref, center, lower_wealth))
                minimum_lower_wealth.append(lower_wealth)
                add_call(
                    path_index,
                    class_index,
                    time,
                    "crossing",
                    math.nan,
                    center,
                    prod,
                    ref,
                    _breakpoints(bets, clip, "crossing"),
                )

            average_minimum = sum(minimum_lower_wealth) / data.samples.shape[2]
            levels = [
                data.samples.shape[2]
                * (1.0 / alpha - average_minimum + minimum_lower_wealth[class_index] / data.samples.shape[2] + padding)
                for class_index in range(data.samples.shape[2])
            ]
            for class_index, (observations, bets, _, _, center, _) in enumerate(crossing_information):
                for call_type in ("lower", "upper"):
                    prod = trace_root(observations, bets, call_type, levels[class_index], center, production)
                    if prod["status"] == 1:
                        continue
                    if prod["status"] != 0:
                        raise RuntimeError(
                            f"endpoint center was infeasible for path={path_index}, class={class_index}, t={time}"
                        )
                    ref = trace_root(observations, bets, call_type, levels[class_index], center, reference)
                    if ref["status"] != 0:
                        raise RuntimeError("reference endpoint unexpectedly skipped the solver")
                    add_call(
                        path_index,
                        class_index,
                        time,
                        call_type,
                        float(levels[class_index]),
                        center,
                        prod,
                        ref,
                        _breakpoints(bets, clip, call_type),
                    )

    if validation_failures:
        raise AssertionError("; ".join(validation_failures[:10]))

    survival = _plot_survival(calls, bracket_map, observation_times)

    summaries = {"overall": _summary(calls)}
    summaries.update(
        {call_type: _summary([row for row in calls if row["call_type"] == call_type]) for call_type in TYPE_LABELS}
    )
    by_time = {
        str(time): _summary([row for row in calls if row["observation_count"] == time])
        for time in observation_times
    }
    summary = {
        "schema_version": 2,
        "experiment": "synthetic TOMS 748 clipping-breakpoint analysis",
        "source_config": "code/experiments/toy/config.yaml",
        "seed": int(toy_cfg.experiment.seed),
        "prediction_paths": int(data.samples.shape[1]),
        "classes": int(data.samples.shape[2]),
        "observation_times": observation_times,
        "clip": clip,
        "production_tolerance": {
            "x": float(production.x_tolerance),
            "y": float(production.y_tolerance),
        },
        "reference_tolerance": {
            "x": float(reference.x_tolerance),
            "y": float(reference.y_tolerance),
        },
        "iteration_convention": "update zero is the initial sign-changing bracket; each later update follows one accepted TOMS 748 proposal",
        "early_termination_convention": "the terminal bracket is carried forward when plotting a fixed denominator; at_risk is reported separately",
        "bootstrap_intervals": "none plotted; calls within a prediction path are correlated",
        "overall": summaries["overall"],
        "by_call_type": {key: summaries[key] for key in TYPE_LABELS},
        "by_observation_count": by_time,
    }
    _write_csv(HERE / "calls.csv.gz", calls)
    _write_csv(HERE / "brackets.csv.gz", brackets)
    _write_csv(HERE / "survival.csv", survival)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_metrics(summary)

    representative_count = int(local_cfg.representative_call_count)
    representatives: list[dict[str, Any]] = []
    candidate_ids: list[int] = []
    for call_type in TYPE_LABELS:
        typed = [row for row in calls if row["call_type"] == call_type]
        clean = [row for row in typed if row["first_clean_update"] is not None]
        censored = [row for row in typed if row["censored"]]
        candidate_ids.append(
            max(clean, key=lambda row: row["first_clean_update"])["call_id"]
        )
        candidate_ids.append((censored or typed)[0]["call_id"])
    for selected_id in list(dict.fromkeys(candidate_ids))[:representative_count]:
        row = calls[selected_id]
        trace = bracket_map[selected_id]
        representatives.append(
            {
                "call": row,
                "initial_bracket": trace[0],
                "terminal_bracket": trace[-1],
                "brackets_nested": True,
                "reference_root_contained_at_every_update": True,
            }
        )
    validation = {
        "calls_checked": len(calls),
        "brackets_checked": len(brackets),
        "failed_checks": validation_failures,
        "checks": [
            "brackets are nested after every accepted proposal",
            "the high-accuracy reference root remains in every production bracket",
            "the high-accuracy reference bracket lies inside the terminal production bracket",
        ],
        "representative_calls": representatives,
    }
    (HERE / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
