from __future__ import annotations

import math

import pytest

from dynnav.evaluation.scientific_metrics import (
    TrialOutcome,
    executed_path_length,
    paired_risk_difference,
    relative_overhead,
    time_integral,
    wilson_interval,
)


def test_irreversible_failure_requires_valid_observed_invalidation() -> None:
    invalid = TrialOutcome(True, False, True, False, True, False)
    assert invalid.irreversible_failure is None
    negative_control = TrialOutcome(True, True, False, False, True, False)
    assert negative_control.irreversible_failure is None


def test_irreversible_failure_separates_mission_and_recovery() -> None:
    irreversible = TrialOutcome(True, True, True, False, True, False)
    recoverable_failure = TrialOutcome(True, True, True, False, True, True)
    success = TrialOutcome(True, True, True, True, True, False)
    assert irreversible.irreversible_failure is True
    assert recoverable_failure.irreversible_failure is False
    assert success.irreversible_failure is False


def test_assessed_recovery_cannot_have_missing_label() -> None:
    outcome = TrialOutcome(True, True, True, False, True, None)
    with pytest.raises(ValueError, match="feasibility"):
        _ = outcome.irreversible_failure


def test_executed_path_length_excludes_reset_jump_when_requested() -> None:
    points = [(0.0, 0.0), (3.0, 4.0), (100.0, 100.0), (103.0, 104.0)]
    assert executed_path_length(points, reset_jump_threshold_m=10.0) == pytest.approx(10.0)
    assert executed_path_length(points) > 100.0


def test_overhead_and_time_integral_have_explicit_units() -> None:
    assert relative_overhead(11.5, 10.0) == pytest.approx(0.15)
    assert time_integral([0.0, 1.0, 3.0], [0.0, 1.0, 1.0]) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="strictly increasing"):
        time_integral([0.0, 0.0], [1.0, 2.0])


def test_wilson_interval_handles_boundary_counts() -> None:
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert 0.0 < high < 0.5
    low, high = wilson_interval(10, 10)
    assert 0.5 < low < 1.0
    assert high == 1.0


def test_paired_risk_difference_retains_discordant_counts() -> None:
    difference, baseline_only, candidate_only = paired_risk_difference(
        [True, True, False, False], [False, True, True, False]
    )
    assert difference == 0.0
    assert baseline_only == 1
    assert candidate_only == 1
    assert math.isfinite(difference)
