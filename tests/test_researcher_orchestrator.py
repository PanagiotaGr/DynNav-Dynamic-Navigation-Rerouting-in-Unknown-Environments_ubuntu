from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dynnav.experiments.end_to_end_evaluation import make_runner
from dynnav.experiments.multiseed_evaluation import TrialRecord
from dynnav.researcher.models import AnalysisPlan, ExperimentSpecification
from dynnav.researcher.orchestrator import ExperimentService


def _specification(seed_count: int = 2) -> ExperimentSpecification:
    return ExperimentSpecification(
        title="Executed canonical comparison",
        research_question="How do all four DynNav objective ablations compare on identical seeded scenarios?",
        hypotheses=["Recoverability awareness may trade path length for preserved escape options."],
        seeds=list(range(seed_count)),
        analysis_plan=AnalysisPlan(bootstrap_resamples=100),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_four_planner_run_generates_auditable_artifacts(tmp_path: Path) -> None:
    service = ExperimentService(tmp_path)
    try:
        status = service.run_sync(_specification())
        assert status.evidence_status == "completed"
        assert status.total_runs == 8
        assert status.completed_runs == 8
        assert status.failed_runs == 0
        assert set(status.summary or {}) == {"shortest", "risk_aware", "recoverability_aware", "proposed"}

        artifact_names = {artifact.filename for artifact in status.artifacts}
        assert artifact_names == {
            "config.json",
            "config.yaml",
            "execution_matrix.json",
            "runs.json",
            "trials.csv",
            "summary.json",
            "report.md",
            "manifest.json",
            "reproducibility_bundle.zip",
        }
        for artifact in status.artifacts:
            path = service.get_artifact_path(status.experiment_id, artifact.filename)
            assert artifact.sha256 == _sha256(path)
            assert artifact.bytes == path.stat().st_size

        manifest = json.loads(service.get_artifact_path(status.experiment_id, "manifest.json").read_text())
        assert manifest["configuration_hash"] == status.configuration_hash
        assert manifest["result_hash"] == _sha256(service.get_artifact_path(status.experiment_id, "trials.csv"))
        assert manifest["planner_ids"] == ["shortest", "risk_aware", "recoverability_aware", "proposed"]

        report = service.get_artifact_path(status.experiment_id, "report.md").read_text()
        assert "executed synthetic software experiment" in report
        assert "do **not** establish ROS 2 or Gazebo validation" in report
        assert "trials.csv" in report
        assert status.configuration_hash in report
    finally:
        service.close()


def test_configured_experiment_exposes_no_numerical_result_block(tmp_path: Path) -> None:
    service = ExperimentService(tmp_path)
    try:
        status = service.register(_specification(seed_count=1))
        assert status.evidence_status == "configured"
        assert status.summary is None
        assert not status.comparisons
        assert not any(artifact.kind in {"results", "report"} for artifact in status.artifacts)
    finally:
        service.close()


def test_partial_run_failure_is_isolated_and_reported(tmp_path: Path) -> None:
    def runner_factory(config):
        real_runner = make_runner(config)

        def runner(seed: int, method: str, parameters: dict[str, float]) -> TrialRecord:
            if seed == 1 and method == "risk_aware":
                raise RuntimeError("injected planner failure")
            return real_runner(seed, method, parameters)

        return runner

    service = ExperimentService(tmp_path, runner_factory=runner_factory)
    try:
        status = service.run_sync(_specification())
        assert status.evidence_status == "partial"
        assert status.completed_runs == 7
        assert status.failed_runs == 1
        assert status.summary is not None
        events = [event for event in status.events if event.event == "run_failed"]
        assert len(events) == 1

        runs = json.loads(service.get_artifact_path(status.experiment_id, "runs.json").read_text())
        failed = [run for run in runs if run["status"] == "failed"]
        assert failed[0]["error"] == "RuntimeError: injected planner failure"
    finally:
        service.close()
