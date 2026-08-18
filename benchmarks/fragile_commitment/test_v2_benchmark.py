"""Scientific-contract tests for the bias-controlled V2 benchmark."""

from __future__ import annotations

from collections import Counter

import pytest

from v2_benchmark import CONDITIONS, EVENT_TARGETS, PLANNERS, V2Condition, run_experiment, run_trial


def test_trial_contains_complete_paired_j0_j3_ablation() -> None:
    rows = run_trial("neutral", 11)
    assert {row["planner"] for row in rows} == set(PLANNERS)
    assert {row["seed"] for row in rows} == {11}
    assert len({row["event_target"] for row in rows}) == 1


def test_event_assignment_is_not_derived_from_selected_route() -> None:
    rows = run_experiment(("neutral",), seeds=300)
    targets = Counter(str(row["event_target"]) for row in rows[:: len(PLANNERS)])
    assert set(targets) == set(EVENT_TARGETS)
    assert all(count > 30 for count in targets.values())
    assert any(row["event_target"] == "resilient" for row in rows)
    assert any(row["event_target"] == "fragile" for row in rows)


def test_risk_direction_is_manipulated_independently() -> None:
    fragile_costly = run_trial("risk_fragile", 4)[0]
    resilient_costly = run_trial("risk_resilient", 4)[0]
    assert fragile_costly["condition"] != resilient_costly["condition"]
    # The same topology/event seed is retained while the risk manipulation flips.
    assert fragile_costly["event_target"] == resilient_costly["event_target"]


def test_invalid_condition_probabilities_are_rejected() -> None:
    condition = V2Condition("open", 0.0, 0.0, event_probabilities=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="sum to one"):
        condition.validate()


def test_all_registered_conditions_validate() -> None:
    for condition in CONDITIONS.values():
        condition.validate()
