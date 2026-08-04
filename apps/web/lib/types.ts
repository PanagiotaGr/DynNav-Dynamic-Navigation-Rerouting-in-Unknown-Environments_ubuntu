export type EvidenceStatus = "configured" | "queued" | "running" | "completed" | "partial" | "failed" | "cancelled";
export type PlannerId = "shortest" | "risk_aware" | "recoverability_aware" | "proposed";

export interface PlannerConfiguration {
  planner_id: PlannerId;
  risk_weight: number;
  irreversibility_weight: number;
  heuristic_weight: number;
  step_cost: number;
}

export interface ExperimentSpecification {
  schema_version: "1.0";
  title: string;
  research_question: string;
  hypotheses: string[];
  scenario: {
    id: string;
    family: "seeded_random_grid";
    description: string;
    parameters: {
      width: number;
      height: number;
      obstacle_probability: number;
      risk_probability: number;
      maximum_risk: number;
    };
    assumptions: string[];
  };
  planners: PlannerConfiguration[];
  seeds: number[];
  metrics: string[];
  analysis_plan: {
    confidence: number;
    bootstrap_resamples: number;
    paired_by_seed: boolean;
    exploratory: boolean;
    significance_claims: boolean;
  };
  requested_outputs: Array<"csv" | "json" | "yaml" | "markdown" | "zip">;
  evidence_status: EvidenceStatus;
  protocol_warnings: string[];
}

export interface ProtocolCompilation {
  specification: ExperimentSpecification;
  tool_call: {
    name: string;
    status: string;
    arguments: Record<string, unknown>;
    result_reference: string;
  };
  capability_notices: string[];
}

export interface Artifact {
  artifact_id: string;
  kind: string;
  filename: string;
  media_type: string;
  sha256: string;
  bytes: number;
  download_url: string;
}

export interface MetricSummary {
  trials: number;
  success_rate: number;
  irreversible_failure_rate: number;
  path_length: { summary: { mean: number }; mean_interval: { lower: number; upper: number; sample_size: number } };
  planning_time_ms: { summary: { mean: number }; mean_interval: { lower: number; upper: number; sample_size: number } };
  cumulative_risk: { summary: { mean: number }; mean_interval: { lower: number; upper: number; sample_size: number } };
  cumulative_irreversibility: { summary: { mean: number }; mean_interval: { lower: number; upper: number; sample_size: number } };
  minimum_escape_options: { summary: { mean: number }; mean_interval: { lower: number; upper: number; sample_size: number } };
}

export interface ProgressEvent {
  sequence: number;
  event: string;
  message: string;
  occurred_at: string;
  completed_runs: number;
  total_runs: number;
  failed_runs: number;
  current_planner: PlannerId | null;
  current_seed: number | null;
}

export interface ExperimentStatus {
  experiment_id: string;
  evidence_status: EvidenceStatus;
  configuration_hash: string;
  completed_runs: number;
  total_runs: number;
  failed_runs: number;
  elapsed_seconds: number;
  current_planner: PlannerId | null;
  current_seed: number | null;
  message: string;
  events: ProgressEvent[];
  artifacts: Artifact[];
  summary: Record<string, MetricSummary> | null;
  comparisons: unknown[];
}

export interface SystemContext {
  git_commit_sha: string;
  git_dirty: boolean | null;
  python_version: string;
  operating_system: string;
  artifact_root: string;
}
