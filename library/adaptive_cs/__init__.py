"""Stateful Python trackers backed by the C++17 confidence-sequence engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np
import torch

from ._cpp import CppBatch as _CppBatch, _FutilityCriterion, build_cpp

EARLY_ABSTENTION_WINDOW = 30
EARLY_ABSTENTION_LATE_OFFSETS = (20, 10, 5)

# ---------------------------------------------------------------------------
# Public configuration and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceSequenceSettings:
    """Settings for the confidence sequence and its numerical inversion."""

    # Confidence-sequence error level.
    alpha: float = 0.05
    # Stake-cap coefficient for K+ and K-.
    c: float = 0.75
    # Initial capital in K+; K- receives 1 - theta.
    theta: float = 0.5
    # Candidate-mean tolerance for root finding.
    x_tolerance: float = 1e-4
    # Wealth-residual tolerance for all root finders.
    y_tolerance: float = 1e-6
    # Maximum iterations per root solve.
    max_steps: int = 80
    # Numerical slack added to bounding-box levels.
    padding: float = 1e-12
    # Intersect bounds with the probability simplex when true.
    simplex_intersection: bool = True
    # Endpoint inversion strategy: full or componentwise.
    endpoint_root_mode: str = "componentwise"
    # Root solver: library Brent--Dekker, Boost.Math TOMS748, or bisection.
    root_finder: str = "boost"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < self.c < 1.0:
            raise ValueError("c must lie in (0, 1)")
        if not 0.0 < self.theta < 1.0:
            raise ValueError("theta must lie in (0, 1)")
        if self.x_tolerance <= 0 or self.y_tolerance <= 0:
            raise ValueError("root-finding tolerances must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.endpoint_root_mode not in {"full", "componentwise"}:
            raise ValueError("endpoint_root_mode must be 'full' or 'componentwise'")
        if self.root_finder not in {"brent", "bisection", "boost"}:
            raise ValueError("root_finder must be 'brent', 'boost', or 'bisection'")


@dataclass(frozen=True)
class UncertaintyEstimate:
    estimate: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

    @property
    def width(self) -> torch.Tensor:
        return self.upper - self.lower


@dataclass(frozen=True)
class CSResult:
    """Immutable snapshot returned by every tracker update."""

    mean: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    prediction: torch.Tensor
    n_samples: torch.Tensor
    resolved: torch.Tensor
    certified: torch.Tensor
    newly_resolved: torch.Tensor
    status: tuple[str, ...]
    decision: Optional[torch.Tensor] = None
    diagnostics: dict[str, int | str | bool] | None = None
    total: Optional[UncertaintyEstimate] = None
    aleatoric: Optional[UncertaintyEstimate] = None
    epistemic: Optional[UncertaintyEstimate] = None


def __iter__(self):
    yield self.mean
    yield self.lower
    yield self.upper
    yield self.certified


# ---------------------------------------------------------------------------
# Python-side tracker logic
# ---------------------------------------------------------------------------


def _as_numpy(values: Any) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)


def _simplex_intersection(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    old_lower, old_upper = lower.copy(), upper.copy()
    lower = np.maximum(
        old_lower, 1.0 - (old_upper.sum(axis=1, keepdims=True) - old_upper)
    )
    upper = np.minimum(
        old_upper, 1.0 - (old_lower.sum(axis=1, keepdims=True) - old_lower)
    )
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.where(probabilities > 0.0, probabilities, 1.0)
    return -(np.where(probabilities > 0.0, probabilities * np.log(safe), 0.0)).sum(
        axis=-1
    )


def _entropy_terms(probabilities: np.ndarray) -> np.ndarray:
    """Return the coordinatewise terms of Shannon entropy."""
    safe = np.where(probabilities > 0.0, probabilities, 1.0)
    return -np.where(probabilities > 0.0, probabilities * np.log(safe), 0.0)


class BoundingBoxTracker:
    """Python lifecycle and stopping logic backed by C++ confidence bounds."""

    criterion = "class"

    def __init__(
        self,
        settings: ConfidenceSequenceSettings | None = None,
        *,
        n_classes: int | None = None,
        alpha: float = 0.05,
        max_samples: int | None = None,
        epsilon: float = 0.05,
        threshold: float = 0.7,
        config: ConfidenceSequenceSettings | None = None,
        stop_on_resolve: bool = True,
        early_abstention: bool = False,
        inspection_times=None,
        **kwargs,
    ):
        if settings is not None and config is not None:
            raise ValueError("pass either settings or config, not both")
        self.config = config or settings or ConfidenceSequenceSettings(alpha=alpha)
        if not isinstance(self.config, ConfidenceSequenceSettings):
            fields = ConfidenceSequenceSettings.__dataclass_fields__
            self.config = ConfidenceSequenceSettings(
                **{
                    name: getattr(self.config, name)
                    for name in fields
                    if hasattr(self.config, name)
                }
            )
        if n_classes is not None and n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        if early_abstention and (max_samples is None or not stop_on_resolve):
            raise ValueError(
                "early_abstention requires max_samples and stop_on_resolve=True"
            )

        if inspection_times is not None and max_samples is None:
            raise ValueError("inspection_times requires max_samples")

        # Normalize inspection_times to a sorted tuple of unique positive integers,
        # and ensure it includes max_samples if specified.
        normalized_inspections = None
        if inspection_times is not None:
            normalized_inspections = tuple(
                sorted({int(value) for value in inspection_times})
            )
            if not normalized_inspections or normalized_inspections[0] < 1:
                raise ValueError("inspection_times must contain positive integers")
            if normalized_inspections[-1] > int(max_samples):
                raise ValueError("inspection_times cannot exceed max_samples")
            if normalized_inspections[-1] != int(max_samples):
                normalized_inspections += (int(max_samples),)

        self.n_classes = n_classes
        self.max_samples = max_samples
        self.epsilon = float(epsilon)
        self.threshold = float(threshold)
        self.stop_on_resolve = bool(stop_on_resolve)
        self.early_abstention = bool(early_abstention)
        self.inspection_times = normalized_inspections
        self._inspection_set = (
            set(normalized_inspections) if normalized_inspections is not None else None
        )
        self._batch_size = None
        self._time = 0
        self._sum = self._lower = self._upper = self._n_samples = self._state = None
        self._entropy_sum = None
        # self._state will have dimension [batch_size] and indicates the current status of each input:
        # 0	unresolved/running
        # 1	resolved/certified
        # 2	empty/infeasible confidence interval
        # 3	abstained because resolution became futile
        self._cpp = None
        self._last_decision = None

    @property
    def _dimension(self) -> int:
        return int(self.n_classes or 0)

    @property
    def _cpp_validates_probabilities(self) -> bool:
        # tells the C++ engine whether each vector should be checked as a probability vector.
        # For Uncertainty this isn't needed and hence overwritten to False.
        return True

    def _encode(self, values: np.ndarray) -> np.ndarray:
        return values

    def _initialize(self, batch_size: int, n_classes: int) -> None:
        if batch_size < 1 or n_classes < 2:
            raise ValueError(
                "the first update must contain a nonempty batch and at least two classes"
            )
        self._batch_size, self.n_classes = batch_size, n_classes
        self._sum = np.zeros((batch_size, n_classes), dtype=np.float64)
        if self.criterion == "uncertainty":
            self._entropy_sum = np.zeros(batch_size, dtype=np.float64)
        self._lower = np.zeros((batch_size, self._dimension), dtype=np.float64)
        self._upper = np.ones_like(self._lower)
        self._n_samples = np.zeros(batch_size, dtype=np.int64)
        self._state = np.zeros(batch_size, dtype=np.int8)
        self._cpp: _CppBatch = _CppBatch(
            batch_size, self.config, validate=self._cpp_validates_probabilities
        )

    def _active_from_indices(self, indices: Any, observed: int) -> np.ndarray:
        assert self._batch_size is not None and self._state is not None
        if indices is None:
            # After a stopping decision, the compact form without an
            # explicit selector follows ``tracker.active_indices``. This keeps a
            # plain ``update(y)`` call from accidentally resampling stopped
            # rows; explicit ``indices`` remains available for arbitrary
            # active subsets.
            active = (
                self.active_indices
                if self._time and self.stop_on_resolve
                else np.arange(self._batch_size, dtype=np.int64)
            )
        elif isinstance(indices, slice):
            active = np.arange(self._batch_size, dtype=np.int64)[indices]
        else:
            raw = np.asarray(indices)
            active = (
                np.flatnonzero(raw)
                if raw.dtype == bool
                else raw.astype(np.int64).reshape(-1)
            )
        if active.size != observed:
            raise ValueError("indices must map every input in the sampled sub-batch")
        if (
            np.any(active < 0)
            or np.any(active >= self._batch_size)
            or len(np.unique(active)) != active.size
        ):
            raise ValueError("indices must be unique and in range")
        if (
            self._time
            and self.stop_on_resolve
            and not np.array_equal(active, self.active_indices)
        ):
            raise ValueError("indices must contain all and only unresolved inputs")
        return active

    def _validate(
        self, values: Any, *, indices: Any = None, check: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if check is False:
            raise ValueError("check=False is not supported by the C++ tracker")
        array = _as_numpy(values)

        if array.ndim == 1:
            array = array[None, None, :]
        elif array.ndim == 2:
            array = array[None, :, :]
        elif array.ndim != 3:
            raise ValueError(
                "values must have shape [class], [input, class], or [draw, input, class]"
            )

        n_draws, observed, n_classes = array.shape

        if self._batch_size is None:
            self._initialize(observed, n_classes)
        assert (
            self._batch_size is not None
            and self._sum is not None
            and self._n_samples is not None
        )

        if n_classes != self.n_classes:
            raise ValueError(f"expected {self.n_classes} classes; got {n_classes}")

        active = self._active_from_indices(indices, observed)

        if not np.isfinite(array).all():
            raise ValueError("predictive probabilities must be finite")
        if (array < -1e-6).any() or (array > 1.0 + 1e-6).any():
            raise ValueError("predictive probabilities must lie in [0, 1]")
        if not np.allclose(array.sum(axis=-1), 1.0, atol=1e-4, rtol=1e-4):
            raise ValueError("predictive probabilities must sum to one")

        return array, active

    def update(
        self, values: Any, *, indices: Any = None, check: bool | None = None
    ) -> CSResult:
        # values can represent:
        # - [class]: one draw for one input
        # - [input, class]: one draw for multiple inputs
        # - [draw, input, class]: multiple draws.
        # in case input, this can also be num_active_inputs so not necessarily entire batch

        # active is a list with indices of active inputs
        array, active = self._validate(
            values, indices=indices, check=check
        )  # [draw, active_inputs, class], [active_inputs]
        n_draws = array.shape[0]
        assert (
            self._batch_size is not None
            and self._sum is not None
            and self._n_samples is not None
        )

        if self.done:  # stop if already finished
            return self._result(np.zeros(self._batch_size, dtype=bool))

        # determine how many draws to consume in this update; respect max_samples
        consume = 0
        if self.max_samples is None:
            consume = n_draws
        else:
            consume = min(n_draws, max(0, self.max_samples - self._time))
        if consume == 0:
            return self._result(np.zeros(self._batch_size, dtype=bool))

        array = array[
            :consume
        ]  # only keep the draws that will be consumed [draw, active_inputs, class]
        assert self._cpp is not None

        if self.inspection_times is None:
            return self._consume(array, active, compute_bounds=True)

        # A scheduled update may cross several inspection times. Consume the
        # supplied draws in order, update the native betting state every time,
        # and reconstruct/check bounds only at the requested absolute times.
        # Inputs that resolve at an inspection are omitted from later draws in
        # the same supplied chunk.
        newly = np.zeros(self._batch_size, dtype=bool)
        column_for_index = {int(index): column for column, index in enumerate(active)}
        for draw in array:
            current_active = (
                self.active_indices if self.stop_on_resolve else active
            )
            if current_active.size == 0:
                break
            columns = np.asarray(
                [column_for_index[int(index)] for index in current_active], dtype=int
            )
            inspect = self._time + 1 in self._inspection_set
            step_result = self._consume(
                draw[columns][None, ...],
                current_active,
                compute_bounds=inspect,
            )
            newly |= step_result.newly_resolved.numpy()
        return self._result(newly)

    def _consume(
        self,
        array: np.ndarray,
        active: np.ndarray,
        *,
        compute_bounds: bool,
    ) -> CSResult:
        """Consume a contiguous draw block and optionally inspect its bounds."""
        assert self._cpp is not None
        assert self._batch_size is not None
        assert self._sum is not None and self._n_samples is not None
        raw_lower, raw_upper = self._cpp.update(
            self._encode(array), active, compute_bounds=compute_bounds
        )
        consumed = array.shape[0]
        self._sum[active] += array.sum(axis=0)
        if self.criterion == "uncertainty":
            assert self._entropy_sum is not None
            self._entropy_sum[active] += _entropy(array).sum(axis=0)
        self._n_samples[active] += consumed
        self._time += consumed

        newly = np.zeros(self._batch_size, dtype=bool)
        if compute_bounds:
            self._lower, self._upper = raw_lower, raw_upper
            if self.criterion == "uncertainty":
                self._lower[:, : self.n_classes], self._upper[:, : self.n_classes] = (
                    _simplex_intersection(
                        self._lower[:, : self.n_classes], self._upper[:, : self.n_classes]
                    )
                )
            newly = self._check_stopping(active)

        # Late futility checks use the updated betting state without requiring
        # another confidence-box inversion. Dense trackers check every draw;
        # sparse trackers also check at three fixed points before the cap.
        if (
            self.early_abstention
            and self._time >= self.max_samples - EARLY_ABSTENTION_WINDOW
            and self._time < self.max_samples
            and (
                self.inspection_times is None
                or compute_bounds
                or self._time in (
                    self.max_samples - offset
                    for offset in EARLY_ABSTENTION_LATE_OFFSETS
                )
            )
        ):
            still_active = self.active_indices
            if still_active.size:
                futile = self._cpp.futile(
                    still_active,
                    self._criterion_code,
                    self.config.x_tolerance,
                    self._futility_level,
                    self.max_samples,
                )
                self._state[still_active[futile]] = 3
                newly[still_active[futile]] = True
        return self._result(newly)

    @property
    def _criterion_code(self) -> int:
        # These values are defined by acs_futility_criterion in the C ABI.
        return int(
            {
                "class": _FutilityCriterion.CLASS,
                "distribution": _FutilityCriterion.DISTRIBUTION,
                "threshold": _FutilityCriterion.THRESHOLD,
            }[self.criterion]
        )

    @property
    def _futility_level(self) -> float:
        return self.threshold if self.criterion == "threshold" else self.epsilon

    def _check_stopping(self, active: np.ndarray) -> np.ndarray:
        assert (
            self._state is not None
            and self._sum is not None
            and self._n_samples is not None
        )
        assert self._lower is not None and self._upper is not None
        previous = self._state.copy()
        rows = np.arange(active.size)
        lower, upper = self._lower[active], self._upper[active]
        empty = (lower > upper).any(axis=1)
        mean = self._sum[active] / np.maximum(self._n_samples[active, None], 1)
        if self.criterion == "class":
            prediction = mean.argmax(axis=1)
            competitors = upper.copy()
            competitors[rows, prediction] = -1.0
            certified = lower[rows, prediction] > competitors.max(axis=1)
            decision = None
        elif self.criterion == "distribution":
            prediction = mean.argmax(axis=1)
            certified = (
                np.maximum(upper - mean, mean - lower).max(axis=1)
                <= self.epsilon + 1e-12
            )
            decision = None
        elif self.criterion == "threshold":
            prediction = mean.argmax(axis=1)
            reached = lower[rows, prediction] >= self.threshold
            certified = reached | (upper.max(axis=1) < self.threshold)
            decision = reached
        else:
            return self._check_uncertainty(active, previous)
        self._state[active] = 0 if not self.stop_on_resolve else self._state[active]
        self._state[active[empty]] = 2
        self._state[active[~empty & certified]] = 1
        if decision is not None:
            if self._last_decision is None:
                self._last_decision = np.zeros(self._batch_size, dtype=bool)
            self._last_decision[active] = decision
        return (self._state != 0) & ~(previous != 0)

    def _check_uncertainty(
        self, active: np.ndarray, previous: np.ndarray
    ) -> np.ndarray:
        quantities = self._uncertainties()
        assert (
            self._state is not None
            and self._lower is not None
            and self._upper is not None
        )
        selected = quantities[self.component]
        lower, upper = selected.lower.numpy()[active], selected.upper.numpy()[active]
        estimate = selected.estimate.numpy()[active]
        if self.stopping == "accuracy":
            certified = np.maximum(upper - estimate, estimate - lower) <= self.epsilon
        else:
            reached = lower >= self.uncertainty_threshold
            certified = reached | (upper < self.uncertainty_threshold)
            if self._last_decision is None:
                self._last_decision = np.zeros(self._batch_size, dtype=bool)
            self._last_decision[active] = reached
        empty = (self._lower[active] > self._upper[active]).any(axis=1)
        empty |= self._lower[active, : self.n_classes].sum(axis=1) > 1 + 1e-12
        empty |= self._upper[active, : self.n_classes].sum(axis=1) < 1 - 1e-12
        for value in quantities.values():
            empty |= value.lower.numpy()[active] > value.upper.numpy()[active]
        self._state[active] = 0
        self._state[active[empty]] = 2
        self._state[active[~empty & certified]] = 1
        return (self._state != 0) & ~(previous != 0)

    def _uncertainties(self) -> dict[str, UncertaintyEstimate]:
        assert self._sum is not None and self._n_samples is not None
        assert self._entropy_sum is not None
        assert self._lower is not None and self._upper is not None
        scale = math.log(self.n_classes)
        mean = self._sum / np.maximum(self._n_samples[:, None], 1)
        # The final coordinate is the augmented per-draw entropy used by the
        # joint confidence sequence; it is not part of the predictive simplex.
        total_estimate = _entropy(mean[:, : self.n_classes])
        aleatoric_estimate = self._entropy_sum / np.maximum(self._n_samples, 1)
        lo, hi = self._lower[:, : self.n_classes], self._upper[:, : self.n_classes]
        total_lower = np.minimum(_entropy_terms(lo), _entropy_terms(hi)).sum(axis=1)
        total_upper = np.minimum(_entropy(np.clip(1.0 / math.e, lo, hi)), scale)
        to_t = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float64))
        total = UncertaintyEstimate(
            to_t(total_estimate), to_t(total_lower), to_t(total_upper)
        )
        aleatoric = UncertaintyEstimate(
            to_t(aleatoric_estimate),
            to_t(self._lower[:, -1] * scale),
            to_t(self._upper[:, -1] * scale),
        )
        epistemic = UncertaintyEstimate(
            to_t(np.clip(total_estimate - aleatoric_estimate, 0, scale)),
            torch.clamp(total.lower - aleatoric.upper, min=0),
            torch.clamp(total.upper - aleatoric.lower, max=scale),
        )
        return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}

    @property
    def active_indices(self):
        if self._state is None:
            return slice(None)
        if self.max_samples is not None and self._time >= self.max_samples:
            return np.empty(0, dtype=np.int64)
        if not self.stop_on_resolve:
            return np.arange(self._batch_size, dtype=np.int64)
        return np.flatnonzero(self._state == 0)

    @property
    def done(self) -> bool:
        if self._state is None:
            return False
        if self.max_samples is not None and self._time >= self.max_samples:
            return True
        return self.stop_on_resolve and self.active_indices.size == 0

    @property
    def remaining(self) -> int:
        return (
            2**31 - 1
            if self.max_samples is None
            else max(0, self.max_samples - self._time)
        )

    @property
    def resolved(self):
        if self._state is None:
            return False
        value = self._state != 0
        return bool(value[0]) if value.size == 1 else value.copy()

    @property
    def all_resolved(self) -> bool:
        return bool(self._state is not None and np.all(self._state != 0))

    @property
    def time(self) -> int:
        return self._time

    @property
    def n_samples(self) -> torch.Tensor:
        if self._n_samples is None:
            raise RuntimeError("n_samples is unavailable before the first update")
        return torch.from_numpy(self._n_samples.copy())

    @property
    def t(self):
        value = self.n_samples
        return int(value[0]) if value.numel() == 1 else value

    @property
    def mean(self) -> torch.Tensor:
        assert self._sum is not None and self._n_samples is not None
        value = torch.from_numpy(
            (self._sum / np.maximum(self._n_samples[:, None], 1)).copy()
        )
        return value[0] if value.shape[0] == 1 else value

    @property
    def lower(self) -> torch.Tensor:
        assert self._lower is not None
        value = torch.from_numpy(self._lower[:, : self.n_classes].copy())
        return value[0] if value.shape[0] == 1 else value

    @property
    def upper(self) -> torch.Tensor:
        assert self._upper is not None
        value = torch.from_numpy(self._upper[:, : self.n_classes].copy())
        return value[0] if value.shape[0] == 1 else value

    @property
    def prediction(self):
        assert self._sum is not None
        value = torch.from_numpy(self._sum.argmax(axis=1).astype(np.int64))
        return int(value[0]) if value.numel() == 1 else value

    @property
    def diagnostics(self):
        return {"approximate": False, "backend": "cpp"}

    def _result(self, newly: np.ndarray) -> CSResult:
        assert (
            self._sum is not None
            and self._lower is not None
            and self._upper is not None
        )
        assert self._state is not None and self._n_samples is not None
        statuses = []
        for value in self._state:
            if value == 1:
                statuses.append("stopped")
            elif value == 2:
                statuses.append("empty")
            elif value == 3:
                statuses.append("abstained")
            elif self.max_samples is not None and self._time >= self.max_samples:
                statuses.append("timeout")
            else:
                statuses.append("running")
        base = CSResult(
            mean=torch.from_numpy(
                (self._sum / np.maximum(self._n_samples[:, None], 1)).copy()
            ),
            lower=torch.from_numpy(self._lower[:, : self.n_classes].copy()),
            upper=torch.from_numpy(self._upper[:, : self.n_classes].copy()),
            prediction=torch.from_numpy(self._sum.argmax(axis=1).astype(np.int64)),
            n_samples=torch.from_numpy(self._n_samples.copy()),
            resolved=torch.from_numpy((self._state != 0)),
            certified=torch.from_numpy((self._state == 1)),
            newly_resolved=torch.from_numpy(newly.copy()),
            status=tuple(statuses),
            decision=(
                None
                if self._last_decision is None
                else torch.from_numpy(self._last_decision.copy())
            ),
            diagnostics=self.diagnostics,
        )
        if self.criterion != "uncertainty":
            return base
        return CSResult(**{**base.__dict__, **self._uncertainties()})


class ClassSelection(BoundingBoxTracker):
    """Stop when the leading class is statistically certified."""

    criterion = "class"


class DistributionAccuracy(BoundingBoxTracker):
    """Stop when every class-probability coordinate is sufficiently precise."""

    criterion = "distribution"


class Threshold(BoundingBoxTracker):
    """Stop when a class probability is certified above or below a threshold."""

    criterion = "threshold"


class Uncertainty(BoundingBoxTracker):
    """C++ uncertainty tracker with accuracy or threshold stopping."""

    criterion = "uncertainty"

    def __init__(
        self,
        *,
        component: str,
        stopping: str,
        epsilon: float | None = None,
        threshold: float | None = None,
        **kwargs,
    ):
        if component not in {"total", "aleatoric", "epistemic"}:
            raise ValueError("component must be 'total', 'aleatoric', or 'epistemic'")
        if stopping not in {"accuracy", "threshold"}:
            raise ValueError(
                "C++ uncertainty stopping must be 'accuracy' or 'threshold'"
            )
        if stopping == "accuracy" and (
            epsilon is None or threshold is not None or epsilon < 0
        ):
            raise ValueError("accuracy stopping requires epsilon only")
        if stopping == "threshold" and (
            threshold is None
            or epsilon is not None
            or threshold < 0
            or not math.isfinite(threshold)
        ):
            raise ValueError(
                "threshold stopping requires a finite nonnegative threshold only"
            )
        if kwargs.get("early_abstention", False):
            raise ValueError("early_abstention is not supported for uncertainty")
        self.component, self.stopping = component, stopping
        self.uncertainty_threshold = threshold
        super().__init__(epsilon=0.0 if epsilon is None else epsilon, **kwargs)

    @property
    def _dimension(self) -> int:
        return int(self.n_classes or 0) + 1

    @property
    def _cpp_validates_probabilities(self) -> bool:
        return False

    def _encode(self, values: np.ndarray) -> np.ndarray:
        normalized = np.clip(_entropy(values) / math.log(self.n_classes), 0.0, 1.0)
        return np.concatenate((values, normalized[..., None]), axis=-1)


__all__ = [
    "ConfidenceSequenceSettings",
    "CSResult",
    "ClassSelection",
    "DistributionAccuracy",
    "Threshold",
    "Uncertainty",
    "UncertaintyEstimate",
    "build_cpp",
]
