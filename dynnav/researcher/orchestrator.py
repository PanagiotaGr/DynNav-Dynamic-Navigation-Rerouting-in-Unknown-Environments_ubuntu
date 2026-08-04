"""Deterministic experiment orchestration and artifact provenance."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import zipfile
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dynnav.experiments.end_to_end_evaluation import make_runner
from dynnav.experiments.multiseed_evaluation import EvaluationConfig, TrialRecord, aggregate, paired_comparisons
from dynnav.experiments.scenario_suite import ScenarioConfig
from dynnav.researcher.models import (
    EvidenceStatus,
    ExecutionMatrix,
    ExperimentArtifact,
    ExperimentRun,
    ExperimentSpecification,
    ExperimentStatus,
    PlannerId,
    ProgressEvent,
    ReproducibilityManifest,
    StatisticalComparison,
)
from dynnav.researcher.reporting import generate_markdown_report

Runner = Callable[[int, str, Mapping[str, float]], TrialRecord]
RunnerFactory = Callable[[ScenarioConfig | None], Runner]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_run_id(
    experiment_id: str, scenario_hash: str, planner_id: PlannerId, seed: int, configuration_hash: str
) -> str:
    payload = f"{experiment_id}:{scenario_hash}:{planner_id}:{seed}:{configuration_hash}"
    return f"run_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def build_execution_matrix(specification: ExperimentSpecification) -> ExecutionMatrix:
    experiment_id = specification.deterministic_id()
    configuration_hash = specification.configuration_hash()
    scenario_hash = specification.scenario.content_hash()
    run_ids = [
        deterministic_run_id(experiment_id, scenario_hash, planner.planner_id, seed, configuration_hash)
        for seed in specification.seeds
        for planner in specification.planners
    ]
    return ExecutionMatrix(
        experiment_id=experiment_id,
        scenario_id=specification.scenario.id,
        planner_ids=[planner.planner_id for planner in specification.planners],
        seeds=list(specification.seeds),
        run_ids=run_ids,
        total_runs=len(run_ids),
    )


@dataclass
class _State:
    specification: ExperimentSpecification
    matrix: ExecutionMatrix
    evidence_status: EvidenceStatus = "configured"
    completed_runs: int = 0
    failed_runs: int = 0
    current_planner: PlannerId | None = None
    current_seed: int | None = None
    message: str = "Experiment configured; execution has not started."
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events: list[ProgressEvent] = field(default_factory=list)
    runs: list[ExperimentRun] = field(default_factory=list)
    artifacts: list[ExperimentArtifact] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    comparisons: list[StatisticalComparison] = field(default_factory=list)
    cancel_requested: bool = False
    future: Future[None] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class ExperimentService:
    """Thread-safe service for configured and executed DynNav experiments."""

    def __init__(
        self,
        artifact_root: str | Path | None = None,
        *,
        repository_root: str | Path | None = None,
        runner_factory: RunnerFactory = make_runner,
        max_workers: int = 2,
    ) -> None:
        default_root = os.environ.get("DYNNAV_ARTIFACT_ROOT", "artifacts/researcher")
        self.artifact_root = Path(artifact_root or default_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.repository_root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
        self.runner_factory = runner_factory
        self._states: dict[str, _State] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dynnav-experiment")

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def register(self, specification: ExperimentSpecification) -> ExperimentStatus:
        experiment_id = specification.deterministic_id()
        with self._lock:
            existing = self._states.get(experiment_id)
            if existing:
                return self._snapshot(existing)

            matrix = build_execution_matrix(specification)
            state = _State(specification=specification, matrix=matrix)
            self._states[experiment_id] = state

        experiment_dir = self._experiment_dir(experiment_id)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(experiment_dir / "config.json", specification.model_dump(mode="json"))
        self._write_text(
            experiment_dir / "config.yaml",
            yaml.safe_dump(specification.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        )
        self._write_json(experiment_dir / "execution_matrix.json", matrix.model_dump(mode="json"))
        state.artifacts = self._artifacts_for(
            experiment_id,
            [
                ("configuration", "config.json", "application/json"),
                ("configuration", "config.yaml", "application/yaml"),
                ("configuration", "execution_matrix.json", "application/json"),
            ],
        )
        self._event(state, "configured", "Created typed experiment configuration.")
        self._persist_state(state)
        return self._snapshot(state)

    def start(self, experiment_id: str) -> ExperimentStatus:
        state = self._require_state(experiment_id)
        with state.lock:
            if state.evidence_status in {"queued", "running"}:
                return self._snapshot(state)
            if state.evidence_status in {"completed", "partial"}:
                raise ValueError("completed experiments are immutable; create a new specification version to rerun")
            state.evidence_status = "queued"
            state.message = "Experiment queued for execution."
            self._event(state, "queued", state.message)
            state.future = self._executor.submit(self._execute, state)
        self._persist_state(state)
        return self._snapshot(state)

    def run_sync(self, specification: ExperimentSpecification) -> ExperimentStatus:
        status = self.register(specification)
        state = self._require_state(status.experiment_id)
        self._execute(state)
        return self._snapshot(state)

    def cancel(self, experiment_id: str) -> ExperimentStatus:
        state = self._require_state(experiment_id)
        with state.lock:
            if state.evidence_status not in {"configured", "queued", "running"}:
                return self._snapshot(state)
            state.cancel_requested = True
            state.message = "Cancellation requested; the active planner call will finish before stopping."
            if state.evidence_status in {"configured", "queued"} and state.future and state.future.cancel():
                state.evidence_status = "cancelled"
                state.finished_at = utcnow()
                self._event(state, "cancelled", "Cancelled queued experiment.")
        self._persist_state(state)
        return self._snapshot(state)

    def get_status(self, experiment_id: str, *, after_sequence: int | None = None) -> ExperimentStatus:
        state = self._require_state(experiment_id)
        snapshot = self._snapshot(state)
        if after_sequence is not None:
            snapshot.events = [event for event in snapshot.events if event.sequence > after_sequence]
        return snapshot

    def get_specification(self, experiment_id: str) -> ExperimentSpecification:
        return self._require_state(experiment_id).specification

    def system_context(self) -> dict[str, Any]:
        commit, dirty = self._git_state()
        return {
            "git_commit_sha": commit,
            "git_dirty": dirty,
            "python_version": platform.python_version(),
            "operating_system": platform.platform(),
            "artifact_root": str(self.artifact_root),
        }

    def get_artifact_path(self, experiment_id: str, filename: str) -> Path:
        state = self._require_state(experiment_id)
        allowed = {artifact.filename for artifact in state.artifacts}
        if filename not in allowed:
            raise FileNotFoundError(filename)
        candidate = (self._experiment_dir(experiment_id) / filename).resolve()
        if candidate.parent != self._experiment_dir(experiment_id).resolve() or not candidate.is_file():
            raise FileNotFoundError(filename)
        return candidate

    def _execute(self, state: _State) -> None:
        with state.lock:
            state.evidence_status = "running"
            state.started_at = utcnow()
            state.message = "Executing deterministic planner matrix."

        spec = state.specification
        params = spec.scenario.parameters
        scenario_config = ScenarioConfig(
            width=params.width,
            height=params.height,
            obstacle_probability=params.obstacle_probability,
            risk_probability=params.risk_probability,
            maximum_risk=params.maximum_risk,
        )
        runner = self.runner_factory(scenario_config)
        records: list[TrialRecord] = []
        run_index = 0

        try:
            for seed in spec.seeds:
                for planner in spec.planners:
                    if state.cancel_requested:
                        self._finish_cancelled(state)
                        return
                    run_id = state.matrix.run_ids[run_index]
                    run_index += 1
                    started_at = utcnow()
                    with state.lock:
                        state.current_planner = planner.planner_id
                        state.current_seed = seed
                        state.message = f"Running {planner.planner_id} for seed {seed}."
                        self._event(state, "run_started", state.message, run_id=run_id)

                    try:
                        record = runner(
                            seed,
                            planner.planner_id,
                            {
                                "risk_weight": planner.risk_weight,
                                "irreversibility_weight": planner.irreversibility_weight,
                                "heuristic_weight": planner.heuristic_weight,
                            },
                        )
                        records.append(record)
                        run = ExperimentRun(
                            run_id=run_id,
                            experiment_id=state.matrix.experiment_id,
                            scenario_id=spec.scenario.id,
                            planner_id=planner.planner_id,
                            seed=seed,
                            status="completed",
                            started_at=started_at,
                            finished_at=utcnow(),
                            configuration_hash=spec.configuration_hash(),
                            outcome="success" if record.success else "failure",
                            failure_reason=None if record.success else "planner returned no feasible path",
                            metrics={
                                "mission_success": record.success,
                                "irreversible_failure": record.irreversible_failure,
                                "path_length": record.path_length,
                                "planning_time_ms": record.planning_time_ms,
                                "nodes_expanded": record.nodes_expanded,
                                "cumulative_risk": record.cumulative_risk,
                                "cumulative_irreversibility": record.cumulative_irreversibility,
                                "minimum_escape_options": record.minimum_escape_options,
                            },
                        )
                        with state.lock:
                            state.runs.append(run)
                            state.completed_runs += 1
                            state.message = f"Completed {state.completed_runs}/{state.matrix.total_runs} runs."
                            self._event(state, "run_completed", state.message, run_id=run_id)
                    except Exception as exc:  # isolate a failed matrix cell and preserve remaining evidence
                        run = ExperimentRun(
                            run_id=run_id,
                            experiment_id=state.matrix.experiment_id,
                            scenario_id=spec.scenario.id,
                            planner_id=planner.planner_id,
                            seed=seed,
                            status="failed",
                            started_at=started_at,
                            finished_at=utcnow(),
                            configuration_hash=spec.configuration_hash(),
                            outcome="error",
                            failure_reason="execution_error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        with state.lock:
                            state.runs.append(run)
                            state.failed_runs += 1
                            state.message = f"Run {run_id} failed; continuing the remaining matrix."
                            self._event(state, "run_failed", state.message, run_id=run_id)
                    self._write_runs(state)
                    self._persist_state(state)

            self._finalize_results(state, records)
        except Exception as exc:
            with state.lock:
                state.evidence_status = "failed"
                state.finished_at = utcnow()
                state.message = f"Experiment orchestration failed: {type(exc).__name__}: {exc}"
                self._event(state, "failed", state.message)
            self._persist_state(state)

    def _finalize_results(self, state: _State, records: list[TrialRecord]) -> None:
        if not records:
            raise RuntimeError("no planner run completed; no numerical summary was generated")
        spec = state.specification
        evaluation_config = EvaluationConfig(
            seeds=tuple(spec.seeds),
            confidence=spec.analysis_plan.confidence,
            bootstrap_resamples=spec.analysis_plan.bootstrap_resamples,
        )
        summary = aggregate(records, config=evaluation_config)
        comparisons: list[StatisticalComparison] = []
        available_methods = {record.method for record in records}
        if "shortest" in available_methods:
            for candidate in ("risk_aware", "recoverability_aware", "proposed"):
                if candidate not in available_methods:
                    continue
                for metric in (
                    "path_length",
                    "planning_time_ms",
                    "cumulative_risk",
                    "cumulative_irreversibility",
                    "minimum_escape_options",
                ):
                    try:
                        result = paired_comparisons(
                            records, "shortest", candidate, metric=metric, config=evaluation_config
                        )
                    except ValueError:
                        continue
                    interval = result["interval"]
                    comparisons.append(
                        StatisticalComparison(
                            baseline="shortest",
                            candidate=candidate,
                            metric=metric,
                            sample_size=interval["sample_size"],
                            mean_difference=result["mean_difference"],
                            standardized_effect=result["standardized_effect"],
                            probability_of_superiority=result["probability_of_superiority"],
                            confidence=interval["confidence"],
                            interval_lower=interval["lower"],
                            interval_upper=interval["upper"],
                        )
                    )

        experiment_id = state.matrix.experiment_id
        experiment_dir = self._experiment_dir(experiment_id)
        self._write_trials(experiment_dir / "trials.csv", records)
        self._write_json(
            experiment_dir / "summary.json",
            {"methods": summary, "paired_comparisons": [item.model_dump(mode="json") for item in comparisons]},
        )
        self._write_runs(state)

        git_commit, git_dirty = self._git_state()
        config_path = os.path.relpath(experiment_dir / "config.yaml", self.repository_root)
        artifact_root = os.path.relpath(self.artifact_root, self.repository_root)
        execution_command = (
            f"python -m dynnav.researcher.cli run " f"--config {config_path} " f"--artifact-root {artifact_root}"
        )
        evidence_references = {
            "config.json": sha256_file(experiment_dir / "config.json"),
            "runs.json": sha256_file(experiment_dir / "runs.json"),
            "summary.json": sha256_file(experiment_dir / "summary.json"),
            "trials.csv": sha256_file(experiment_dir / "trials.csv"),
        }
        report = generate_markdown_report(
            experiment_id=experiment_id,
            specification=spec,
            summary=summary,
            comparisons=comparisons,
            runs=state.runs,
            git_commit_sha=git_commit,
            git_dirty=git_dirty,
            execution_command=execution_command,
            evidence_references=evidence_references,
        )
        self._write_text(experiment_dir / "report.md", report)

        artifact_specs: list[tuple[str, str, str]] = [
            ("configuration", "config.json", "application/json"),
            ("configuration", "config.yaml", "application/yaml"),
            ("configuration", "execution_matrix.json", "application/json"),
            ("runs", "runs.json", "application/json"),
            ("results", "trials.csv", "text/csv"),
            ("results", "summary.json", "application/json"),
            ("report", "report.md", "text/markdown"),
        ]
        artifacts_before_manifest = self._artifacts_for(experiment_id, artifact_specs)
        result_hash = sha256_file(experiment_dir / "trials.csv")
        manifest = ReproducibilityManifest(
            experiment_id=experiment_id,
            configuration_hash=spec.configuration_hash(),
            scenario_hash=spec.scenario.content_hash(),
            result_hash=result_hash,
            git_commit_sha=git_commit,
            git_dirty=git_dirty,
            python_version=platform.python_version(),
            operating_system=platform.platform(),
            dependency_versions=self._dependency_versions(),
            execution_command=execution_command,
            seed_list=list(spec.seeds),
            planner_ids=[planner.planner_id for planner in spec.planners],
            generated_at=utcnow(),
            artifacts=artifacts_before_manifest,
        )
        self._write_json(experiment_dir / "manifest.json", manifest.model_dump(mode="json"))
        artifact_specs.append(("manifest", "manifest.json", "application/json"))
        self._write_bundle(experiment_dir)
        artifact_specs.append(("bundle", "reproducibility_bundle.zip", "application/zip"))
        artifacts = self._artifacts_for(experiment_id, artifact_specs)

        with state.lock:
            state.summary = summary
            state.comparisons = comparisons
            state.artifacts = artifacts
            state.evidence_status = "partial" if state.failed_runs else "completed"
            state.finished_at = utcnow()
            state.current_planner = None
            state.current_seed = None
            state.message = (
                "Experiment completed with partial evidence because one or more runs failed."
                if state.failed_runs
                else "Experiment completed; results and provenance artifacts are available."
            )
            self._event(
                state,
                "completed",
                state.message,
                artifact_ids=[artifact.artifact_id for artifact in artifacts],
            )
        self._persist_state(state)

    def _finish_cancelled(self, state: _State) -> None:
        with state.lock:
            state.evidence_status = "cancelled"
            state.finished_at = utcnow()
            state.current_planner = None
            state.current_seed = None
            state.message = "Experiment cancelled; no aggregate result was generated."
            self._event(state, "cancelled", state.message)
        self._write_runs(state)
        self._persist_state(state)

    def _event(
        self,
        state: _State,
        event: str,
        message: str,
        *,
        run_id: str | None = None,
        artifact_ids: list[str] | None = None,
    ) -> None:
        state.events.append(
            ProgressEvent(
                sequence=len(state.events) + 1,
                event=event,
                occurred_at=utcnow(),
                message=message,
                completed_runs=state.completed_runs,
                total_runs=state.matrix.total_runs,
                failed_runs=state.failed_runs,
                current_planner=state.current_planner,
                current_seed=state.current_seed,
                run_id=run_id,
                artifact_ids=artifact_ids or [],
            )
        )

    def _snapshot(self, state: _State) -> ExperimentStatus:
        with state.lock:
            started = state.started_at or state.created_at
            finished = state.finished_at or utcnow()
            elapsed = max(0.0, (finished - started).total_seconds()) if state.started_at else 0.0
            return ExperimentStatus(
                experiment_id=state.matrix.experiment_id,
                evidence_status=state.evidence_status,
                configuration_hash=state.specification.configuration_hash(),
                completed_runs=state.completed_runs,
                total_runs=state.matrix.total_runs,
                failed_runs=state.failed_runs,
                elapsed_seconds=elapsed,
                current_planner=state.current_planner,
                current_seed=state.current_seed,
                message=state.message,
                events=list(state.events),
                artifacts=list(state.artifacts),
                summary=state.summary,
                comparisons=list(state.comparisons),
            )

    def _require_state(self, experiment_id: str) -> _State:
        with self._lock:
            try:
                return self._states[experiment_id]
            except KeyError as exc:
                raise KeyError(f"unknown experiment: {experiment_id}") from exc

    def _experiment_dir(self, experiment_id: str) -> Path:
        if not experiment_id.startswith("exp_") or not experiment_id[4:].isalnum():
            raise ValueError("invalid experiment identifier")
        return self.artifact_root / experiment_id

    def _artifacts_for(
        self, experiment_id: str, specifications: list[tuple[str, str, str]]
    ) -> list[ExperimentArtifact]:
        artifacts: list[ExperimentArtifact] = []
        for kind, filename, media_type in specifications:
            path = self._experiment_dir(experiment_id) / filename
            digest = sha256_file(path)
            artifacts.append(
                ExperimentArtifact(
                    artifact_id=f"art_{digest[:16]}",
                    kind=kind,
                    filename=filename,
                    media_type=media_type,
                    sha256=digest,
                    bytes=path.stat().st_size,
                    download_url=f"/v1/experiments/{experiment_id}/artifacts/{filename}",
                )
            )
        return artifacts

    def _write_runs(self, state: _State) -> None:
        self._write_json(
            self._experiment_dir(state.matrix.experiment_id) / "runs.json",
            [run.model_dump(mode="json") for run in state.runs],
        )

    def _persist_state(self, state: _State) -> None:
        snapshot = self._snapshot(state)
        self._write_json(
            self._experiment_dir(state.matrix.experiment_id) / "state.json",
            snapshot.model_dump(mode="json"),
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        ExperimentService._write_text(path, text + "\n")

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _write_trials(path: Path, records: list[TrialRecord]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
        temporary.replace(path)

    @staticmethod
    def _write_bundle(experiment_dir: Path) -> None:
        target = experiment_dir / "reproducibility_bundle.zip"
        temporary = target.with_suffix(".zip.tmp")
        names = (
            "config.json",
            "config.yaml",
            "execution_matrix.json",
            "runs.json",
            "trials.csv",
            "summary.json",
            "report.md",
            "manifest.json",
        )
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(experiment_dir / name, arcname=name)
        temporary.replace(target)

    def _git_state(self) -> tuple[str, bool | None]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.repository_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            )
            return commit, dirty
        except (OSError, subprocess.SubprocessError):
            return "unavailable", None

    @staticmethod
    def _dependency_versions() -> dict[str, str]:
        versions: dict[str, str] = {}
        for name in ("dynnav", "numpy", "pydantic", "PyYAML", "scipy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "source-checkout" if name == "dynnav" else "unavailable"
        return versions
