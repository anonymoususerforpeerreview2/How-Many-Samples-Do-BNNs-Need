import numpy as np

from non_vectorized_class_selection import (
    ClassSelection,
    ConfidenceSequenceSettings,
    scalar_wealth_path,
)


def test_scalar_wealth_path_matches_class_selection_component() -> None:
    settings = ConfidenceSequenceSettings(alpha=0.05, c=0.08, theta=0.5)
    rng = np.random.default_rng(417)
    observations = rng.beta(34.0, 16.0, size=60)
    path = scalar_wealth_path(observations, candidate=0.5, settings=settings)
    tracker = ClassSelection(settings)

    internal_wealth = []
    for observation in observations:
        tracker.update(np.array([observation, 1.0 - observation]))
        history = np.stack(tracker._history)[:, 0]
        bets = np.stack(tracker._bets)[:, 0]
        internal_wealth.append(tracker._wealth(history, bets, candidate=0.5))

    np.testing.assert_allclose(path.wealth, internal_wealth, rtol=1e-14, atol=0.0)


def test_scalar_wealth_path_records_predictable_state() -> None:
    settings = ConfidenceSequenceSettings(alpha=0.05, c=0.08, theta=0.5)
    observations = np.array([0.65, 0.72, 0.61])
    path = scalar_wealth_path(observations, candidate=0.5, settings=settings)

    assert path.wealth.shape == observations.shape
    assert path.positive.shape == observations.shape
    assert path.negative.shape == observations.shape
    assert path.unconstrained_bets.shape == observations.shape
    assert path.variance_numerator >= 0.25
    assert np.all(path.wealth > 0.0)


def test_reference_boxes_are_intersected_over_time() -> None:
    rng = np.random.default_rng(29)
    tracker = ClassSelection()
    previous_lower = np.zeros(3)
    previous_upper = np.ones(3)

    for draw in rng.dirichlet((3.0, 2.0, 1.0), size=25):
        result = tracker.update(draw)
        assert np.all(result.lower >= previous_lower - 1e-12)
        assert np.all(result.upper <= previous_upper + 1e-12)
        previous_lower, previous_upper = result.lower, result.upper
