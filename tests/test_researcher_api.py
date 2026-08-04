from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from dynnav.researcher.api import create_app
from dynnav.researcher.orchestrator import ExperimentService


def test_api_protocol_execution_results_and_report_contract(tmp_path: Path) -> None:
    service = ExperimentService(tmp_path)
    client = TestClient(create_app(service))
    try:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["evidence_policy"] == "executed-artifacts-only"

        compiled = client.post(
            "/v1/protocols",
            json={"request": "Compare all four planners over 1 deterministic seed and generate a reproducible report."},
        )
        assert compiled.status_code == 200
        specification = compiled.json()["specification"]
        assert specification["evidence_status"] == "configured"
        assert len(specification["planners"]) == 4

        created = client.post("/v1/experiments", json=specification)
        assert created.status_code == 201
        experiment_id = created.json()["experiment_id"]
        assert created.json()["summary"] is None

        unavailable = client.get(f"/v1/experiments/{experiment_id}/results")
        assert unavailable.status_code == 409

        queued = client.post(f"/v1/experiments/{experiment_id}/run")
        assert queued.status_code == 202
        deadline = time.monotonic() + 15
        payload = queued.json()
        while payload["evidence_status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.05)
            payload = client.get(f"/v1/experiments/{experiment_id}").json()

        assert payload["evidence_status"] == "completed"
        assert payload["completed_runs"] == 4
        assert payload["summary"] is not None
        report = next(item for item in payload["artifacts"] if item["filename"] == "report.md")

        download = client.get(report["download_url"])
        assert download.status_code == 200
        assert download.headers["content-type"].startswith("text/markdown")
        assert "Executed results" in download.text
    finally:
        service.close()


def test_api_rejects_unknown_experiment_and_unsafe_artifact_path(tmp_path: Path) -> None:
    service = ExperimentService(tmp_path)
    client = TestClient(create_app(service))
    try:
        assert client.get("/v1/experiments/exp_missing").status_code == 404
        assert client.get("/v1/experiments/exp_missing/artifacts/config.json").status_code == 404
    finally:
        service.close()
