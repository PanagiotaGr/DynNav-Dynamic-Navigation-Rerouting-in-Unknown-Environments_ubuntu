"""Markdown reporting from executed DynNav artifacts only."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dynnav.researcher.models import ExperimentRun, ExperimentSpecification, StatisticalComparison

PLANNER_NAMES = {
    "shortest": "Shortest path",
    "risk_aware": "Risk-aware",
    "recoverability_aware": "Recoverability-aware",
    "proposed": "Joint risk + recoverability",
}


def _number(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def generate_markdown_report(
    *,
    experiment_id: str,
    specification: ExperimentSpecification,
    summary: dict[str, dict[str, Any]],
    comparisons: Iterable[StatisticalComparison],
    runs: Iterable[ExperimentRun],
    git_commit_sha: str,
    git_dirty: bool | None,
    execution_command: str,
    evidence_references: dict[str, str],
) -> str:
    completed_runs = [run for run in runs if run.status == "completed"]
    failed_runs = [run for run in runs if run.status == "failed"]
    lines = [
        f"# {specification.title}",
        "",
        "> Evidence status: **executed synthetic software experiment**. All numerical values in this report ",
        "> were computed from the referenced DynNav run artifacts. Interpretation is explicitly labelled.",
        "",
        "## Abstract",
        "",
        f"This report records a paired four-planner comparison over {len(specification.seeds)} deterministic seeds "
        f"({len(completed_runs)} completed planner runs; {len(failed_runs)} execution errors). The experiment compares "
        "geometric, risk-aware, recoverability-aware, and joint objectives on identical seeded scenarios.",
        "",
        "## Research question",
        "",
        specification.research_question,
        "",
        "## Hypotheses",
        "",
    ]
    lines.extend(f"- {hypothesis}" for hypothesis in specification.hypotheses)
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"- Experiment ID: `{experiment_id}`",
            f"- Scenario family: `{specification.scenario.family}` (`{specification.scenario.id}`)",
            f"- Scenario dimensions: {specification.scenario.parameters.width} × "
            f"{specification.scenario.parameters.height} cells",
            f"- Seeds: {', '.join(str(seed) for seed in specification.seeds)}",
            f"- Planned runs: {len(specification.seeds) * len(specification.planners)}",
            f"- Bootstrap confidence: {specification.analysis_plan.confidence:.1%}",
            f"- Bootstrap resamples: {specification.analysis_plan.bootstrap_resamples}",
            "- Comparison design: paired by identical scenario seed and generator configuration",
            "- Analysis status: exploratory; no automatic statistical-significance claim",
            "",
            "### Planner objectives",
            "",
            "| Planner | λ risk | λ irreversibility | Objective |",
            "|---|---:|---:|---|",
        ]
    )
    for planner in specification.planners:
        name = PLANNER_NAMES[planner.planner_id]
        objective = "L"
        if planner.risk_weight:
            objective += " + λ risk × R"
        if planner.irreversibility_weight:
            objective += " + λ irr × I"
        lines.append(f"| {name} | {planner.risk_weight:g} | {planner.irreversibility_weight:g} | {objective} |")

    lines.extend(
        [
            "",
            "Risk and irreversibility are normalized to `[0, 1]` at the state level by the current planner "
            "implementation; "
            "the weights are dimensionless trade-off coefficients.",
            "",
            "## Executed results",
            "",
            "| Planner | n | Success | Irreversible failure | Mean path length | Mean risk | Mean irreversibility | "
            "Mean planning time (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for planner in specification.planners:
        row = summary.get(planner.planner_id)
        if not row:
            lines.append(f"| {PLANNER_NAMES[planner.planner_id]} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            "| {name} | {trials} | {success:.1%} | {failure:.1%} | {path} | {risk} | {irr} | {time} |".format(
                name=PLANNER_NAMES[planner.planner_id],
                trials=row["trials"],
                success=row["success_rate"],
                failure=row["irreversible_failure_rate"],
                path=_number(row["path_length"]["summary"]["mean"]),
                risk=_number(row["cumulative_risk"]["summary"]["mean"]),
                irr=_number(row["cumulative_irreversibility"]["summary"]["mean"]),
                time=_number(row["planning_time_ms"]["summary"]["mean"]),
            )
        )

    lines.extend(["", "## Exploratory paired comparisons", ""])
    comparisons = list(comparisons)
    if comparisons:
        lines.extend(
            [
                "Differences are candidate minus shortest-path baseline. Negative values therefore indicate a lower "
                "candidate value.",
                "",
                "| Candidate | Metric | n | Mean difference | Bootstrap interval | Standardized effect | "
                "P(candidate < baseline) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in comparisons:
            lines.append(
                f"| {PLANNER_NAMES[comparison.candidate]} | `{comparison.metric}` | {comparison.sample_size} | "
                f"{_number(comparison.mean_difference)} | [{_number(comparison.interval_lower)}, "
                f"{_number(comparison.interval_upper)}] | {_number(comparison.standardized_effect)} | "
                f"{comparison.probability_of_superiority:.1%} |"
            )
    else:
        lines.append("No complete paired samples were available.")

    lines.extend(["", "## Execution failures", ""])
    if failed_runs:
        lines.extend(
            f"- `{run.run_id}` — {PLANNER_NAMES[run.planner_id]}, seed {run.seed}: {run.error or 'unknown error'}"
            for run in failed_runs
        )
    else:
        lines.append("No orchestration errors were recorded.")

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {assumption}" for assumption in specification.scenario.assumptions)
    if specification.protocol_warnings:
        lines.extend(f"- {warning}" for warning in specification.protocol_warnings)
    lines.append(
        "- This slice evaluates one static plan per seeded grid; it does not yet model online obstacle events, "
        "replanning, recovery actions, or emergency stops."
    )

    lines.extend(["", "## Evidence references", ""])
    lines.extend(f"- `{filename}` — SHA-256 `{digest}`" for filename, digest in sorted(evidence_references.items()))
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The tables above are observations from the executed synthetic grid experiments. The hypotheses are not "
            "treated as confirmed merely because a point estimate changes. Any scientific interpretation must consider "
            "the bootstrap interval, "
            "effect size, paired sample count, scenario limitations, and execution failures.",
            "",
            "These results do **not** establish ROS 2 or Gazebo validation, physical-robot performance, "
            "causal effects, "
            "formal safety "
            "guarantees, or deployment readiness.",
            "",
            "## Reproducibility",
            "",
            f"- Git commit: `{git_commit_sha}`",
            f"- Working tree dirty at execution: `{git_dirty}`",
            f"- Configuration hash: `{specification.configuration_hash()}`",
            f"- Reproduction command: `{execution_command}`",
            "",
            "The accompanying manifest records the Python version, operating system, dependency versions, "
            "scenario hash, "
            "result hash, "
            "and artifact checksums.",
            "",
        ]
    )
    return "\n".join(lines)
