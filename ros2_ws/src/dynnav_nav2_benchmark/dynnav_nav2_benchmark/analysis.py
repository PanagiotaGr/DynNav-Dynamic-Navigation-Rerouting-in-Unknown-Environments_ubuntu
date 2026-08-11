"""ROS-independent contracts and metrics for the Nav2 benchmark runner."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class Pose2D:
    """Planar pose in a named metric coordinate frame."""

    x: float
    y: float
    yaw: float = 0.0

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.yaw)):
            raise ValueError("pose coordinates must be finite")


@dataclass(frozen=True, slots=True)
class PlannerSpec:
    """Planner identity and declared objective terms."""

    planner_id: str
    family: str
    risk_weight: float | None = None
    irreversibility_weight: float | None = None

    def validate(self) -> None:
        if not self.planner_id or not self.family:
            raise ValueError("planner_id and family must be non-empty")
        for value in (self.risk_weight, self.irreversibility_weight):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError("planner weights must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """One paired start/goal query evaluated by every planner."""

    name: str
    start: Pose2D
    goal: Pose2D
    frame_id: str = "map"

    def validate(self) -> None:
        if not self.name or not self.frame_id:
            raise ValueError("scenario name and frame_id must be non-empty")
        self.start.validate()
        self.goal.validate()
        if math.hypot(self.goal.x - self.start.x, self.goal.y - self.start.y) <= 0.0:
            raise ValueError("scenario start and goal must differ")


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """Versioned paired-query benchmark definition."""

    schema_version: int
    seed: int
    planners: tuple[PlannerSpec, ...]
    scenarios: tuple[ScenarioSpec, ...]

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.planners or not self.scenarios:
            raise ValueError("benchmark requires planners and scenarios")
        planner_ids = [planner.planner_id for planner in self.planners]
        scenario_names = [scenario.name for scenario in self.scenarios]
        if len(set(planner_ids)) != len(planner_ids):
            raise ValueError("planner IDs must be unique")
        if len(set(scenario_names)) != len(scenario_names):
            raise ValueError("scenario names must be unique")
        for planner in self.planners:
            planner.validate()
        for scenario in self.scenarios:
            scenario.validate()


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Raw result from one planner-server request."""

    scenario: str
    planner_id: str
    repetition: int
    order_index: int
    success: bool
    planning_latency_ms: float
    path_length_m: float | None
    pose_count: int
    goal_error_m: float | None
    path_xy: tuple[tuple[float, float], ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pose_from_mapping(payload: dict[str, Any]) -> Pose2D:
    return Pose2D(
        x=float(payload["x"]),
        y=float(payload["y"]),
        yaw=float(payload.get("yaw", 0.0)),
    )


def load_suite(path: str | Path) -> BenchmarkSuite:
    """Load and validate a benchmark-suite YAML file."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark suite must be a YAML mapping")
    planners = tuple(
        PlannerSpec(
            planner_id=str(item["planner_id"]),
            family=str(item["family"]),
            risk_weight=(
                float(item["risk_weight"]) if "risk_weight" in item else None
            ),
            irreversibility_weight=(
                float(item["irreversibility_weight"])
                if "irreversibility_weight" in item
                else None
            ),
        )
        for item in payload.get("planners", [])
    )
    scenarios = tuple(
        ScenarioSpec(
            name=str(item["name"]),
            frame_id=str(item.get("frame_id", "map")),
            start=_pose_from_mapping(item["start"]),
            goal=_pose_from_mapping(item["goal"]),
        )
        for item in payload.get("scenarios", [])
    )
    suite = BenchmarkSuite(
        schema_version=int(payload.get("schema_version", 0)),
        seed=int(payload.get("seed", 0)),
        planners=planners,
        scenarios=scenarios,
    )
    suite.validate()
    return suite


def balanced_trial_order(
    planner_ids: Sequence[str], repetitions: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic complete blocks with counterbalanced positions."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not planner_ids or len(set(planner_ids)) != len(planner_ids):
        raise ValueError("planner_ids must be a non-empty unique sequence")
    blocks: list[tuple[str, ...]] = []
    planner_count = len(planner_ids)
    for cycle_start in range(0, repetitions, planner_count):
        base = list(planner_ids)
        random.Random(seed + cycle_start // planner_count).shuffle(base)
        cycle_length = min(planner_count, repetitions - cycle_start)
        for offset in range(cycle_length):
            blocks.append(tuple(base[offset:] + base[:offset]))
    return tuple(blocks)


def path_length(points: Sequence[tuple[float, float]]) -> float:
    """Return Euclidean polyline length in meters."""

    return sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points, points[1:], strict=False)
    )


def summarize_trials(records: Iterable[TrialRecord]) -> dict[str, dict[str, Any]]:
    """Aggregate raw records without discarding failures."""

    grouped: dict[str, list[TrialRecord]] = {}
    for record in records:
        if record.planning_latency_ms < 0.0:
            raise ValueError("planning latency cannot be negative")
        grouped.setdefault(record.planner_id, []).append(record)

    summary: dict[str, dict[str, Any]] = {}
    for planner_id, rows in sorted(grouped.items()):
        successful = [row for row in rows if row.success]
        latencies = [row.planning_latency_ms for row in rows]
        lengths = [
            row.path_length_m
            for row in successful
            if row.path_length_m is not None
        ]
        goal_errors = [
            row.goal_error_m
            for row in successful
            if row.goal_error_m is not None
        ]
        summary[planner_id] = {
            "trials": len(rows),
            "successes": len(successful),
            "failures": len(rows) - len(successful),
            "success_rate": len(successful) / len(rows),
            "planning_latency_ms_mean": statistics.fmean(latencies),
            "planning_latency_ms_median": statistics.median(latencies),
            "planning_latency_ms_p95": _linear_percentile(latencies, 0.95),
            "planning_latency_ms_stddev_population": statistics.pstdev(latencies),
            "path_length_m_mean": statistics.fmean(lengths) if lengths else None,
            "path_length_m_median": statistics.median(lengths) if lengths else None,
            "goal_error_m_mean": (
                statistics.fmean(goal_errors) if goal_errors else None
            ),
        }
    return summary


def summarize_trials_by_scenario(
    records: Iterable[TrialRecord],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return scenario-stratified summaries to prevent hidden pooling effects."""

    grouped: dict[str, list[TrialRecord]] = {}
    for record in records:
        grouped.setdefault(record.scenario, []).append(record)
    return {
        scenario: summarize_trials(rows)
        for scenario, rows in sorted(grouped.items())
    }


def resolve_map_image(map_yaml: str | Path) -> Path:
    """Resolve and validate the occupancy image referenced by a Nav2 map YAML."""

    source = Path(map_yaml).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("image"), str):
        raise ValueError("map YAML must define an image path")
    image = Path(payload["image"])
    if not image.is_absolute():
        image = source.parent / image
    image = image.resolve()
    if not image.is_file():
        raise ValueError(f"map image does not exist: {image}")
    return image


def _linear_percentile(values: Sequence[float], fraction: float) -> float:
    """Compute a deterministic linearly interpolated percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
