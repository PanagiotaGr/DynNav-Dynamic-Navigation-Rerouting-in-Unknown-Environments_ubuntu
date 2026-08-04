"""Typed contracts shared by the DynNav researcher service and web client.

The models deliberately separate a configured protocol from executed evidence.
Numerical result models are populated only by the experiment orchestrator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PlannerId = Literal["shortest", "risk_aware", "recoverability_aware", "proposed"]
EvidenceStatus = Literal["configured", "queued", "running", "completed", "partial", "failed", "cancelled"]
RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchToolCall(StrictModel):
    name: str
    status: Literal["proposed", "confirmed", "running", "completed", "failed"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_reference: str | None = None


class ResearchMessage(StrictModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    created_at: datetime
    evidence_status: EvidenceStatus | Literal["interpretation", "hypothesis", "unsupported"]
    tool_calls: list[ResearchToolCall] = Field(default_factory=list)


class ResearchSession(StrictModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    archived: bool = False
    messages: list[ResearchMessage] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)


class ScenarioParameters(StrictModel):
    width: Annotated[int, Field(ge=4, le=100)] = 20
    height: Annotated[int, Field(ge=4, le=100)] = 14
    obstacle_probability: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.18
    risk_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    maximum_risk: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9


class ScenarioDefinition(StrictModel):
    id: str = "seeded-random-grid-v1"
    family: Literal["seeded_random_grid"] = "seeded_random_grid"
    description: str = (
        "Deterministic synthetic grid with a guaranteed start-to-goal corridor, "
        "seeded obstacles, risk, and uncertainty layers."
    )
    parameters: ScenarioParameters = Field(default_factory=ScenarioParameters)
    assumptions: list[str] = Field(
        default_factory=lambda: [
            "Software simulation only",
            "Static occupancy during each individual planning run",
            "Risk and irreversibility are normalized to [0, 1] per state",
        ]
    )

    def content_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PlannerDefinition(StrictModel):
    id: PlannerId
    name: str
    objective: str
    source: str = "dynnav.planners.recoverability_astar"


class PlannerConfiguration(StrictModel):
    planner_id: PlannerId
    risk_weight: Annotated[float, Field(ge=0.0, le=1000.0)]
    irreversibility_weight: Annotated[float, Field(ge=0.0, le=1000.0)]
    heuristic_weight: Annotated[float, Field(gt=0.0, le=10.0)] = 1.0
    step_cost: Annotated[float, Field(gt=0.0, le=1000.0)] = 1.0

    @model_validator(mode="after")
    def validate_ablation(self) -> PlannerConfiguration:
        if self.planner_id == "shortest" and (self.risk_weight or self.irreversibility_weight):
            raise ValueError("shortest must set both objective weights to zero")
        if self.planner_id == "risk_aware" and self.irreversibility_weight:
            raise ValueError("risk_aware must set irreversibility_weight to zero")
        if self.planner_id == "recoverability_aware" and self.risk_weight:
            raise ValueError("recoverability_aware must set risk_weight to zero")
        return self


class MetricDefinition(StrictModel):
    id: str
    name: str
    unit: str
    direction: Literal["higher_is_better", "lower_is_better", "descriptive"]
    definition: str


class AnalysisPlan(StrictModel):
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.95
    bootstrap_resamples: Annotated[int, Field(ge=100, le=100_000)] = 2000
    paired_by_seed: bool = True
    exploratory: bool = True
    significance_claims: bool = False


def default_planners(risk_weight: float = 4.0, irreversibility_weight: float = 4.0) -> list[PlannerConfiguration]:
    return [
        PlannerConfiguration(planner_id="shortest", risk_weight=0.0, irreversibility_weight=0.0),
        PlannerConfiguration(planner_id="risk_aware", risk_weight=risk_weight, irreversibility_weight=0.0),
        PlannerConfiguration(
            planner_id="recoverability_aware", risk_weight=0.0, irreversibility_weight=irreversibility_weight
        ),
        PlannerConfiguration(
            planner_id="proposed", risk_weight=risk_weight, irreversibility_weight=irreversibility_weight
        ),
    ]


DEFAULT_METRICS = [
    "mission_success_rate",
    "irreversible_failure_rate",
    "path_length",
    "planning_time_ms",
    "cumulative_risk",
    "cumulative_irreversibility",
    "minimum_escape_options",
]


class ExperimentSpecification(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    title: Annotated[str, Field(min_length=3, max_length=160)]
    research_question: Annotated[str, Field(min_length=10, max_length=4000)]
    hypotheses: Annotated[list[str], Field(min_length=1, max_length=12)]
    scenario: ScenarioDefinition = Field(default_factory=ScenarioDefinition)
    planners: Annotated[list[PlannerConfiguration], Field(min_length=1, max_length=4)] = Field(
        default_factory=default_planners
    )
    seeds: Annotated[list[int], Field(min_length=1, max_length=500)] = Field(default_factory=lambda: list(range(30)))
    metrics: Annotated[list[str], Field(min_length=1)] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    analysis_plan: AnalysisPlan = Field(default_factory=AnalysisPlan)
    requested_outputs: list[Literal["csv", "json", "yaml", "markdown", "zip"]] = Field(
        default_factory=lambda: ["csv", "json", "yaml", "markdown"]
    )
    evidence_status: EvidenceStatus = "configured"
    protocol_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_protocol(self) -> ExperimentSpecification:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if any(seed < 0 or seed > 2_147_483_647 for seed in self.seeds):
            raise ValueError("seeds must be in [0, 2147483647]")
        planner_ids = [planner.planner_id for planner in self.planners]
        if len(set(planner_ids)) != len(planner_ids):
            raise ValueError("planner configurations must be unique")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be unique")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("evidence_status", None)
        return payload

    def configuration_hash(self) -> str:
        canonical = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def deterministic_id(self) -> str:
        return f"exp_{self.configuration_hash()[:16]}"


class ExecutionMatrix(StrictModel):
    experiment_id: str
    scenario_id: str
    planner_ids: list[PlannerId]
    seeds: list[int]
    run_ids: list[str]
    total_runs: int
    fairness_key: str = "identical scenario generator, seed, start, goal, and risk realization per planner"


class MetricResult(StrictModel):
    metric_id: str
    value: float | bool | str
    unit: str
    evidence_reference: str


class ExperimentRun(StrictModel):
    run_id: str
    experiment_id: str
    scenario_id: str
    planner_id: PlannerId
    seed: int
    status: RunStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    configuration_hash: str
    outcome: Literal["success", "failure", "error", "cancelled"] | None = None
    failure_reason: str | None = None
    metrics: dict[str, float | bool] = Field(default_factory=dict)
    error: str | None = None


class StatisticalComparison(StrictModel):
    baseline: PlannerId
    candidate: PlannerId
    metric: str
    sample_size: int
    mean_difference: float
    standardized_effect: float
    probability_of_superiority: float
    confidence: float
    interval_lower: float
    interval_upper: float
    exploratory: bool = True


class ExperimentArtifact(StrictModel):
    artifact_id: str
    kind: Literal["configuration", "runs", "results", "report", "manifest", "log", "bundle"]
    filename: str
    media_type: str
    sha256: str
    bytes: int
    download_url: str


class GeneratedReport(StrictModel):
    artifact: ExperimentArtifact
    evidence_references: list[str]
    generated_at: datetime


class ReproducibilityManifest(StrictModel):
    experiment_id: str
    configuration_hash: str
    scenario_hash: str
    result_hash: str
    git_commit_sha: str
    git_dirty: bool | None
    python_version: str
    operating_system: str
    dependency_versions: dict[str, str]
    execution_command: str
    seed_list: list[int]
    planner_ids: list[PlannerId]
    generated_at: datetime
    artifacts: list[ExperimentArtifact] = Field(default_factory=list)


class ProgressEvent(StrictModel):
    sequence: int
    event: Literal[
        "configured", "queued", "run_started", "run_completed", "run_failed", "cancelled", "completed", "failed"
    ]
    occurred_at: datetime
    message: str
    completed_runs: int
    total_runs: int
    failed_runs: int = 0
    current_planner: PlannerId | None = None
    current_seed: int | None = None
    run_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)


class ExperimentStatus(StrictModel):
    experiment_id: str
    evidence_status: EvidenceStatus
    configuration_hash: str
    completed_runs: int
    total_runs: int
    failed_runs: int
    elapsed_seconds: float
    current_planner: PlannerId | None = None
    current_seed: int | None = None
    message: str
    events: list[ProgressEvent] = Field(default_factory=list)
    artifacts: list[ExperimentArtifact] = Field(default_factory=list)
    summary: dict[str, Any] | None = None
    comparisons: list[StatisticalComparison] = Field(default_factory=list)


class NaturalLanguageProtocolRequest(StrictModel):
    request: Annotated[str, Field(min_length=10, max_length=8000)]


class ProtocolCompilation(StrictModel):
    specification: ExperimentSpecification
    tool_call: ResearchToolCall
    capability_notices: list[str]
