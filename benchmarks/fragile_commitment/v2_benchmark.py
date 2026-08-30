"""Bias-controlled J0--J3 route-choice benchmark.

The original benchmark is intentionally a positive-control counterexample: its
closure is always placed on the route labelled ``fragile``.  This module is the
scientific V2 instrument.  Route properties and event targets are controlled
independently, and all four objectives use the same candidate set.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark import evaluate_route
from topology_families import TopologyConfig, generate_topology

PLANNERS = ("J0_shortest", "J1_risk", "J2_recoverability", "J3_joint")
EVENT_TARGETS = ("none", "fragile", "resilient")


@dataclass(frozen=True, slots=True)
class V2Condition:
    """One preregisterable manipulation of route costs and event exposure."""

    family: str
    fragile_risk_bias: float
    resilient_risk_bias: float
    event_probabilities: tuple[float, float, float] = (0.2, 0.4, 0.4)
    risk_weight: float = 20.0
    recoverability_weight: float = 20.0

    def validate(self) -> None:
        if self.family not in {"open", "bottleneck", "culdesac", "multiroute"}:
            raise ValueError(f"unknown topology family: {self.family}")
        if min(self.fragile_risk_bias, self.resilient_risk_bias) < 0.0:
            raise ValueError("risk biases must be non-negative")
        if self.risk_weight < 0.0 or self.recoverability_weight < 0.0:
            raise ValueError("objective weights must be non-negative")
        if len(self.event_probabilities) != len(EVENT_TARGETS):
            raise ValueError("event probabilities must cover none/fragile/resilient")
        if any(value < 0.0 for value in self.event_probabilities):
            raise ValueError("event probabilities must be non-negative")
        if not np.isclose(sum(self.event_probabilities), 1.0):
            raise ValueError("event probabilities must sum to one")


CONDITIONS: dict[str, V2Condition] = {
    "neutral": V2Condition("open", 0.0, 0.0),
    "risk_fragile": V2Condition("open", 0.03, 0.0),
    "risk_resilient": V2Condition("open", 0.0, 0.03),
    "recovery_dominant": V2Condition("bottleneck", 0.0, 0.0),
    "risk_recovery_conflict": V2Condition("bottleneck", 0.0, 0.03),
    "deceptive_local_degree": V2Condition("culdesac", 0.0, 0.0),
}


def _select_event(seed: int, probabilities: tuple[float, float, float]) -> str:
    # A separate RNG stream prevents risk-field draws from changing event labels.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD1A5]))
    return str(rng.choice(EVENT_TARGETS, p=probabilities))


def _objective(planner: str, row: dict[str, object], condition: V2Condition) -> float:
    length = float(row["path_length"])
    risk = float(row["route_risk"])
    fragility = float(row["fragility_penalty"])
    if planner == "J0_shortest":
        return length
    if planner == "J1_risk":
        return length + condition.risk_weight * risk
    if planner == "J2_recoverability":
        return length + condition.recoverability_weight * fragility
    if planner == "J3_joint":
        return length + condition.risk_weight * risk + condition.recoverability_weight * fragility
    raise ValueError(f"unknown planner: {planner}")


def run_trial(condition_name: str, seed: int) -> list[dict[str, object]]:
    """Return four paired planner rows for one independently randomized world."""

    try:
        condition = CONDITIONS[condition_name]
    except KeyError as exc:
        raise ValueError(f"unknown V2 condition: {condition_name}") from exc
    condition.validate()
    topology = TopologyConfig(
        family=condition.family,
        fragile_risk_bias=condition.fragile_risk_bias,
        resilient_risk_bias=condition.resilient_risk_bias,
    )
    scenario = generate_topology(seed, topology)
    candidates = {
        "fragile": evaluate_route(scenario, "fragile", scenario.fragile_route),
        "resilient": evaluate_route(scenario, "resilient", scenario.resilient_route),
    }
    event_target = _select_event(seed, condition.event_probabilities)
    rows: list[dict[str, object]] = []
    for planner in PLANNERS:
        selected = min(
            candidates.values(),
            key=lambda row: (_objective(planner, row, condition), str(row["route"])),
        )
        route = str(selected["route"])
        event_blocks_route = event_target == route
        rows.append(
            {
                "condition": condition_name,
                "family": condition.family,
                "seed": seed,
                "planner": planner,
                "event_target": event_target,
                **selected,
                "event_blocks_route": event_blocks_route,
                "mission_success": not event_blocks_route,
                "objective_value": _objective(planner, selected, condition),
            }
        )
    return rows


def run_experiment(condition_names: tuple[str, ...], seeds: int) -> list[dict[str, object]]:
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    return [
        row
        for condition_name in condition_names
        for seed in range(seeds)
        for row in run_trial(condition_name, seed)
    ]


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        raise ValueError("cannot export an empty experiment")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--output", type=Path, default=Path("v2_results.csv"))
    args = parser.parse_args()
    rows = run_experiment(tuple(args.conditions), args.seeds)
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} paired J0--J3 rows to {args.output}")


if __name__ == "__main__":
    main()
