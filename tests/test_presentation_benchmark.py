from __future__ import annotations

import argparse
import math

import pytest

from scripts.presentation_benchmark import aggregate, parse_seeds


def test_parse_seeds_accepts_ranges_and_removes_duplicates() -> None:
    assert parse_seeds("1-3,3,7") == [1, 2, 3, 7]


def test_parse_seeds_rejects_descending_range() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seeds("5-2")


def test_aggregate_reports_mean_and_population_std() -> None:
    rows = [
        {
            "planner": "A*",
            "static_success": 1,
            "static_path_cells": 10,
            "static_expansions": 20,
            "static_runtime_ms": 1.0,
            "static_cost": 9.0,
            "static_avg_risk": 0.4,
            "static_max_risk": 0.8,
            "rollout_success": 1,
            "rollout_distance": 10.0,
            "rollout_replans": 2,
            "rollout_avg_risk": 0.3,
            "rollout_max_risk": 0.7,
            "rollout_avg_compute_ms": 1.2,
            "rollout_collisions": 1,
        },
        {
            "planner": "A*",
            "static_success": 0,
            "static_path_cells": 0,
            "static_expansions": 30,
            "static_runtime_ms": 3.0,
            "static_cost": float("nan"),
            "static_avg_risk": 1.0,
            "static_max_risk": 1.0,
            "rollout_success": 0,
            "rollout_distance": 6.0,
            "rollout_replans": 4,
            "rollout_avg_risk": 0.5,
            "rollout_max_risk": 0.9,
            "rollout_avg_compute_ms": 2.2,
            "rollout_collisions": 3,
        },
    ]

    summary = aggregate(rows)

    assert len(summary) == 1
    result = summary[0]
    assert result["planner"] == "A*"
    assert result["runs"] == 2
    assert result["rollout_success_mean"] == pytest.approx(0.5)
    assert result["rollout_distance_mean"] == pytest.approx(8.0)
    assert result["rollout_distance_std"] == pytest.approx(2.0)
    assert result["static_cost_mean"] == pytest.approx(9.0)
    assert math.isfinite(float(result["static_cost_mean"]))
