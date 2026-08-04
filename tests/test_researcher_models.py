from __future__ import annotations

import pytest
from pydantic import ValidationError

from dynnav.researcher.models import (
    ExperimentSpecification,
    PlannerConfiguration,
)
from dynnav.researcher.orchestrator import build_execution_matrix, deterministic_run_id
from dynnav.researcher.protocols import compile_research_request


def _specification(**overrides) -> ExperimentSpecification:
    payload = {
        "title": "Canonical planner comparison",
        "research_question": "How do the four canonical planners compare under paired seeded scenarios?",
        "hypotheses": ["Planner objectives may produce different path and risk trade-offs."],
        "seeds": [2, 5, 9],
    }
    payload.update(overrides)
    return ExperimentSpecification(**payload)


def test_protocol_compiler_extracts_seed_count_weights_and_capability_boundary() -> None:
    compilation = compile_research_request(
        "Compare all four planners in a dynamic environment over 12 seeds with lambda_risk = 2.5 and lambda_irr = 7."
    )
    spec = compilation.specification

    assert len(spec.seeds) == 12
    assert spec.planners[1].risk_weight == 2.5
    assert spec.planners[2].irreversibility_weight == 7.0
    assert spec.evidence_status == "configured"
    assert any("dynamic obstacle events" in notice for notice in compilation.capability_notices)
    assert "planner_count" in compilation.tool_call.arguments


def test_configuration_id_is_deterministic_and_ignores_only_evidence_state() -> None:
    configured = _specification(evidence_status="configured")
    completed = configured.model_copy(update={"evidence_status": "completed"})

    assert configured.configuration_hash() == completed.configuration_hash()
    assert configured.deterministic_id() == completed.deterministic_id()
    assert configured.deterministic_id().startswith("exp_")


def test_specification_rejects_duplicate_seeds_and_invalid_ablation_weights() -> None:
    with pytest.raises(ValidationError, match="seeds must be unique"):
        _specification(seeds=[1, 1])

    with pytest.raises(ValidationError, match="shortest must set both objective weights to zero"):
        PlannerConfiguration(
            planner_id="shortest",
            risk_weight=1.0,
            irreversibility_weight=0.0,
        )

    with pytest.raises(ValidationError, match="risk_aware must set irreversibility_weight to zero"):
        PlannerConfiguration(
            planner_id="risk_aware",
            risk_weight=1.0,
            irreversibility_weight=1.0,
        )


def test_execution_matrix_pairs_every_planner_with_identical_seed_set() -> None:
    specification = _specification()
    matrix = build_execution_matrix(specification)

    assert matrix.total_runs == len(specification.seeds) * len(specification.planners)
    assert len(matrix.run_ids) == len(set(matrix.run_ids))
    assert matrix.seeds == specification.seeds
    assert set(matrix.planner_ids) == {"shortest", "risk_aware", "recoverability_aware", "proposed"}

    first = deterministic_run_id(
        specification.deterministic_id(),
        specification.scenario.content_hash(),
        "shortest",
        specification.seeds[0],
        specification.configuration_hash(),
    )
    assert first == matrix.run_ids[0]
