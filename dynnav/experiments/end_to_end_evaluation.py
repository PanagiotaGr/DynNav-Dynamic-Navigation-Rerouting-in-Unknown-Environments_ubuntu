"""End-to-end evaluation runner for all recoverability A* ablations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from dynnav.experiments.multiseed_evaluation import (
    EvaluationConfig,
    TrialRecord,
    aggregate,
    paired_comparisons,
    run_multiseed,
    write_artifacts,
)
from dynnav.experiments.scenario_suite import ScenarioConfig, generate_scenario
from dynnav.planners.recoverability_astar import (
    PlannerMode,
    RecoverabilityAStarConfig,
    recoverability_astar,
)

DEFAULT_METHODS: tuple[str, ...] = tuple(mode.value for mode in PlannerMode)


@dataclass(frozen=True)
class ExperimentResult:
    records: tuple[TrialRecord, ...]
    summary: dict[str, dict]
    comparisons: dict[str, dict]


def make_runner(scenario_config: ScenarioConfig | None = None):
    cfg = scenario_config or ScenarioConfig()
    cfg.validate()

    def runner(seed: int, method: str, parameters: Mapping[str, float]) -> TrialRecord:
        try:
            mode = PlannerMode(method)
        except ValueError as exc:
            raise ValueError(f"unknown planner method: {method}") from exc
        scenario = generate_scenario(seed, cfg)
        planner_config = RecoverabilityAStarConfig(
            risk_weight=float(parameters.get("risk_weight", 4.0)),
            irreversibility_weight=float(parameters.get("irreversibility_weight", 4.0)),
            heuristic_weight=float(parameters.get("heuristic_weight", 1.0)),
        )
        result = recoverability_astar(
            scenario.grid,
            scenario.start,
            scenario.goal,
            safe_cells=set(scenario.safe_cells),
            mode=mode,
            config=planner_config,
        )
        return TrialRecord(
            seed=seed,
            method=method,
            success=result.success,
            irreversible_failure=not result.success,
            path_length=float(result.geometric_length),
            planning_time_ms=float(result.planning_time_ms),
            nodes_expanded=float(result.nodes_expanded),
            cumulative_risk=float(result.cumulative_risk),
            cumulative_irreversibility=float(result.cumulative_irreversibility),
            minimum_escape_options=float(result.minimum_escape_options),
        )

    return runner


def run_experiment(
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    evaluation_config: EvaluationConfig | None = None,
    scenario_config: ScenarioConfig | None = None,
    parameters: Mapping[str, float] | None = None,
    output_dir: str | Path | None = None,
) -> ExperimentResult:
    config = evaluation_config or EvaluationConfig()
    records = run_multiseed(
        make_runner(scenario_config), methods, parameters=parameters, config=config
    )
    summary = aggregate(records, config=config)
    comparisons: dict[str, dict] = {}
    if PlannerMode.SHORTEST.value in methods and PlannerMode.PROPOSED.value in methods:
        for metric in (
            "path_length",
            "planning_time_ms",
            "nodes_expanded",
            "cumulative_risk",
            "cumulative_irreversibility",
            "minimum_escape_options",
        ):
            comparisons[metric] = paired_comparisons(
                records,
                PlannerMode.SHORTEST.value,
                PlannerMode.PROPOSED.value,
                metric=metric,
                config=config,
            )
    artifact_summary = {"methods": summary, "shortest_vs_proposed": comparisons}
    if output_dir is not None:
        write_artifacts(records, artifact_summary, output_dir)
    return ExperimentResult(tuple(records), summary, comparisons)
