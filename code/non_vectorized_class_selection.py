"""Small, readable class-selection reference based on ordinary bisection.

This module is intentionally independent of the native tracker package.  It
provides the readable timing baseline used in the paper as well as reusable
scalar wealth diagnostics for paper figures.  It supports one probability
stream at a time and is not intended to be a second production implementation
of the library.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ConfidenceSequenceSettings:
    alpha: float = 0.05
    c: float = 0.75
    theta: float = 0.5
    x_tolerance: float = 1e-4
    y_tolerance: float = 1e-6
    max_steps: int = 80
    padding: float = 1e-12


@dataclass(frozen=True)
class Result:
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class ScalarWealthPath:
    """Directional and hedged wealth values for one scalar observation stream."""

    wealth: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    unconstrained_bets: np.ndarray
    variance_numerator: float


def unconstrained_bet(
    alpha: float,
    variance_numerator: float | np.ndarray,
    time: int,
) -> np.ndarray:
    """Return the predictable stake from Equation (12).

    ``variance_numerator`` is ``time * sigma_hat_{time-1}^2``, equivalently
    ``1/4`` plus the squared regularized-mean deviations observed before the
    current draw.  Arrays are accepted for the coordinate-wise class-selection
    update; a scalar produces a zero-dimensional array.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if time < 1:
        raise ValueError("time must be positive")
    variance_numerator = np.asarray(variance_numerator, dtype=float)
    if np.any(~np.isfinite(variance_numerator)) or np.any(
        variance_numerator < 0.0
    ):
        raise ValueError("variance_numerator must be finite and nonnegative")
    numerator = 2.0 * math.log(2.0 / alpha)
    denominator = variance_numerator * math.log1p(time)
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def clipped_bets(
    bets: float | np.ndarray,
    candidate: float,
    clip: float,
    component: int,
) -> np.ndarray:
    """Clip unconstrained bets for the positive or negative wealth component."""
    if not 0.0 <= candidate <= 1.0:
        raise ValueError("candidate must lie in [0, 1]")
    if not 0.0 < clip < 1.0:
        raise ValueError("clip must lie strictly between zero and one")
    if component not in (0, 1):
        raise ValueError("component must be 0 (positive) or 1 (negative)")
    denominator = candidate if component == 0 else 1.0 - candidate
    cap = math.inf if denominator == 0.0 else clip / denominator
    return np.minimum(np.asarray(bets, dtype=float), cap)


def scalar_wealth_path(
    observations: np.ndarray,
    candidate: float,
    settings: ConfidenceSequenceSettings | None = None,
) -> ScalarWealthPath:
    """Evaluate the reference hedged wealth along one scalar stream.

    This is the scalar counterpart of the capital calculation used internally
    by :class:`ClassSelection`.  It is useful for diagnostics and figures that
    need the complete wealth path rather than only the inverted confidence
    bounds.
    """
    settings = settings or ConfidenceSequenceSettings()
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1:
        raise ValueError("observations must be a one-dimensional scalar stream")
    if (
        np.any(~np.isfinite(observations))
        or np.any(observations < 0.0)
        or np.any(observations > 1.0)
    ):
        raise ValueError("observations must be finite values in [0, 1]")
    if not 0.0 <= candidate <= 1.0:
        raise ValueError("candidate must lie in [0, 1]")

    positive = 1.0
    negative = 1.0
    variance_numerator = 0.25
    running_sum = 0.0
    wealth_values = np.empty(observations.size, dtype=float)
    positive_values = np.empty_like(wealth_values)
    negative_values = np.empty_like(wealth_values)
    bets = np.empty_like(wealth_values)

    for index, observation in enumerate(observations, start=1):
        bet = float(unconstrained_bet(settings.alpha, variance_numerator, index))
        positive_bet = float(
            clipped_bets(bet, candidate, settings.c, component=0)
        )
        negative_bet = float(
            clipped_bets(bet, candidate, settings.c, component=1)
        )

        positive *= 1.0 + positive_bet * (observation - candidate)
        negative *= 1.0 - negative_bet * (observation - candidate)
        positive_values[index - 1] = positive
        negative_values[index - 1] = negative
        wealth_values[index - 1] = max(
            settings.theta * positive,
            (1.0 - settings.theta) * negative,
        )
        bets[index - 1] = bet

        running_sum += observation
        regularized_mean = (0.5 + running_sum) / (index + 1.0)
        variance_numerator += (observation - regularized_mean) ** 2

    return ScalarWealthPath(
        wealth=wealth_values,
        positive=positive_values,
        negative=negative_values,
        unconstrained_bets=bets,
        variance_numerator=variance_numerator,
    )


class ClassSelection:
    """A simple non-vectorized class-selection confidence sequence."""

    def __init__(self, settings: ConfidenceSequenceSettings | None = None) -> None:
        self.settings = settings or ConfidenceSequenceSettings()
        self._history: list[np.ndarray] = []
        self._bets: list[np.ndarray] = []
        self._sum = None
        self._sum_x = None
        self._sum_dev = None
        self._variance = None
        self._result: Result | None = None

    def update(self, value: np.ndarray) -> Result:
        value = np.asarray(value, dtype=float)
        if value.ndim != 1 or value.size < 2:
            raise ValueError("value must be a one-dimensional probability vector")
        if not np.all(np.isfinite(value)) or np.any(value < 0) or not np.isclose(value.sum(), 1.0):
            raise ValueError("value must be a finite probability vector summing to one")
        if self._sum is None:
            self._sum = np.zeros_like(value)
            self._sum_x = np.zeros_like(value)
            self._sum_dev = np.zeros_like(value)
            self._variance = np.full_like(value, 0.25)
        elif value.shape != self._sum.shape:
            raise ValueError("the number of classes cannot change")

        t = len(self._history) + 1
        self._sum += value
        bet = unconstrained_bet(self.settings.alpha, self._variance * t, t)
        regularized_mean = (0.5 + self._sum_x + value) / (t + 1.0)
        self._sum_dev += (value - regularized_mean) ** 2
        self._sum_x += value
        self._variance = (0.25 + self._sum_dev) / (t + 1.0)
        self._history.append(value.copy())
        self._bets.append(bet)
        result = self._bounds()
        if self._result is not None:
            result = Result(
                lower=np.maximum(result.lower, self._result.lower),
                upper=np.minimum(result.upper, self._result.upper),
            )
        self._result = result
        return result

    def _log_component(self, history: np.ndarray, bets: np.ndarray, candidate: float, component: int) -> float:
        signed_bets = clipped_bets(bets, candidate, self.settings.c, component)
        if component == 1:
            signed_bets = -signed_bets
        weight = self.settings.theta if component == 0 else 1.0 - self.settings.theta
        return math.log(weight) + float(np.log1p(signed_bets * (history - candidate)).sum())

    def _log_wealth(self, history: np.ndarray, bets: np.ndarray, candidate: float) -> float:
        return max(
            self._log_component(history, bets, candidate, component)
            for component in (0, 1)
        )

    def _wealth(self, history: np.ndarray, bets: np.ndarray, candidate: float) -> float:
        return math.exp(min(700.0, self._log_wealth(history, bets, candidate)))

    def _minimum(self, history: np.ndarray, bets: np.ndarray) -> tuple[float, float, float]:
        left, right = 0.0, 1.0
        left_components = [math.exp(min(700.0, self._log_component(history, bets, left, c))) for c in (0, 1)]
        right_components = [math.exp(min(700.0, self._log_component(history, bets, right, c))) for c in (0, 1)]
        left_logs = [self._log_component(history, bets, left, c) for c in (0, 1)]
        right_logs = [self._log_component(history, bets, right, c) for c in (0, 1)]
        boundary = left_logs[1] > left_logs[0] or right_logs[1] < right_logs[0]
        if boundary:
            left_wealth = max(left_components)
            right_wealth = max(right_components)
            return min(left_wealth, right_wealth), (0.0 if left_wealth <= right_wealth else 1.0), 0.0

        for _ in range(self.settings.max_steps):
            x_converged = right - left <= self.settings.x_tolerance
            y_converged = (
                min(right_components[1], left_components[0])
                - max(left_components[1], right_components[0])
                <= self.settings.y_tolerance
            )
            if x_converged and y_converged:
                break
            midpoint = 0.5 * (left + right)
            midpoint_logs = [self._log_component(history, bets, midpoint, c) for c in (0, 1)]
            midpoint_components = [math.exp(min(700.0, value)) for value in midpoint_logs]
            if midpoint_logs[1] < midpoint_logs[0]:
                left, left_components = midpoint, midpoint_components
            elif midpoint_logs[1] > midpoint_logs[0]:
                right, right_components = midpoint, midpoint_components
            else:
                left = right = midpoint
                left_components = right_components = midpoint_components
            if (
                right - left <= self.settings.x_tolerance
                and min(right_components[1], left_components[0])
                - max(left_components[1], right_components[0])
                <= self.settings.y_tolerance
            ):
                break
        minimum_lower = max(left_components[1], right_components[0])
        return minimum_lower, 0.5 * (left + right), 0.0

    def _endpoint(self, history: np.ndarray, bets: np.ndarray, level: float, center: float, lower: bool) -> float:
        outside, inside = ((0.0, center) if lower else (1.0, center))
        if self._wealth(history, bets, outside) <= level:
            return outside
        outside_residual = math.log(level) - self._log_wealth(history, bets, outside)
        inside_residual = math.log(level) - self._log_wealth(history, bets, inside)
        for _ in range(self.settings.max_steps):
            if (
                abs(inside - outside) <= self.settings.x_tolerance
                and min(abs(outside_residual), abs(inside_residual))
                <= self.settings.y_tolerance
            ):
                break
            midpoint = 0.5 * (outside + inside)
            midpoint_residual = math.log(level) - self._log_wealth(history, bets, midpoint)
            if midpoint_residual >= 0.0:
                inside = midpoint
                inside_residual = midpoint_residual
            else:
                outside = midpoint
                outside_residual = midpoint_residual
        return outside

    def _bounds(self) -> Result:
        history = np.stack(self._history)
        bets = np.stack(self._bets)
        classes = history.shape[1]
        minima = []
        centers = []
        for cls in range(classes):
            minimum, center, _ = self._minimum(history[:, cls], bets[:, cls])
            minima.append(minimum)
            centers.append(center)
        minima = np.asarray(minima)
        levels = classes * (1.0 / self.settings.alpha - minima.mean() + minima / classes + self.settings.padding)
        lower = np.empty(classes)
        upper = np.empty(classes)
        for cls in range(classes):
            center = centers[cls]
            if self._wealth(history[:, cls], bets[:, cls], center) > levels[cls]:
                lower[cls], upper[cls] = 1.0, 0.0
            else:
                lower[cls] = self._endpoint(history[:, cls], bets[:, cls], levels[cls], center, True)
                upper[cls] = self._endpoint(history[:, cls], bets[:, cls], levels[cls], center, False)
        return Result(lower=lower, upper=upper)


if __name__ == "__main__":
    sequence = ClassSelection()
    for _ in range(3):
        print(sequence.update(np.array([0.8, 0.15, 0.05])))
