"""Transparent natural-language to experiment-protocol compilation.

This first vertical slice uses a deterministic compiler instead of asking a
language model to invent a protocol or any result. A future model integration
can call the same typed service after explicit user confirmation.
"""

from __future__ import annotations

import re

from dynnav.researcher.models import (
    AnalysisPlan,
    ExperimentSpecification,
    ProtocolCompilation,
    ResearchToolCall,
    ScenarioDefinition,
    default_planners,
)

DEFAULT_REQUEST = (
    "Compare shortest-path, risk-aware, recoverability-aware, and joint planning "
    "over 30 deterministic seeds. Measure irreversible failure rate, path length, "
    "cumulative risk, minimum escape options, and planning time."
)


def _read_integer(patterns: list[str], text: str, default: int) -> int:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return default


def _read_float(patterns: list[str], text: str, default: float) -> float:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return default


def compile_research_request(request: str) -> ProtocolCompilation:
    cleaned = " ".join(request.strip().split())
    if len(cleaned) < 10:
        raise ValueError("research request must contain at least 10 characters")

    seed_count = _read_integer(
        [r"(?:use|over|with|across)\s+(\d+)\s+(?:deterministic\s+)?seeds?", r"(\d+)\s+seeds?"],
        cleaned,
        30,
    )
    seed_count = max(1, min(seed_count, 500))
    risk_weight = _read_float(
        [r"lambda[_\s-]*risk\s*(?:=|of|to)?\s*(\d+(?:\.\d+)?)", r"risk weight\s*(?:=|of|to)?\s*(\d+(?:\.\d+)?)"],
        cleaned,
        4.0,
    )
    irreversibility_weight = _read_float(
        [
            r"lambda[_\s-]*(?:irr|irreversibility)\s*(?:=|of|to)?\s*(\d+(?:\.\d+)?)",
            r"irreversibility weight\s*(?:=|of|to)?\s*(\d+(?:\.\d+)?)",
        ],
        cleaned,
        4.0,
    )

    lowered = cleaned.lower()
    warnings: list[str] = []
    unsupported_dynamic = any(term in lowered for term in ("dynamic", "moving obstacle", "route closure"))
    if unsupported_dynamic:
        warnings.append(
            "The current executable slice uses a static seeded grid per run; dynamic obstacle events "
            "remain unsupported and are not represented as executed evidence."
        )
    if any(term in lowered for term in ("replan", "recovery success", "emergency stop")):
        warnings.append(
            "Recovery success, emergency-stop rate, replan count, and replanning time require an online event "
            "simulation and are not emitted by the current static planner slice."
        )
    if "gazebo" in lowered or "ros 2" in lowered or "ros2" in lowered or "robot" in lowered:
        warnings.append(
            "This protocol produces synthetic software-simulation evidence only; it does not validate ROS 2, "
            "Gazebo, physical robots, or safety guarantees."
        )

    hypotheses = [
        "Recoverability-aware objectives may reduce irreversible failures relative to shortest-path planning.",
        "Any reduction in irreversible failure may trade off against path length and planning time.",
        "Joint risk-and-recoverability planning may improve risk exposure while preserving escape options.",
    ]
    title = "Four-planner recoverability comparison"
    if "lambda" in lowered or "weight" in lowered:
        title = "Planner objective-weight comparison"

    specification = ExperimentSpecification(
        title=title,
        research_question=cleaned,
        hypotheses=hypotheses,
        scenario=ScenarioDefinition(),
        planners=default_planners(risk_weight, irreversibility_weight),
        seeds=list(range(seed_count)),
        analysis_plan=AnalysisPlan(),
        protocol_warnings=warnings,
    )
    return ProtocolCompilation(
        specification=specification,
        tool_call=ResearchToolCall(
            name="create_experiment",
            status="completed",
            arguments={
                "planner_count": len(specification.planners),
                "seed_count": len(specification.seeds),
                "scenario_family": specification.scenario.family,
            },
            result_reference=specification.deterministic_id(),
        ),
        capability_notices=[
            "Protocol compilation is deterministic and does not execute an experiment.",
            "Numerical result blocks remain unavailable until the configured experiment completes.",
            *warnings,
        ],
    )
