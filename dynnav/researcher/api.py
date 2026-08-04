"""FastAPI surface for the DynNav Researcher vertical slice."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from dynnav.researcher.models import (
    ExperimentSpecification,
    ExperimentStatus,
    NaturalLanguageProtocolRequest,
    ProtocolCompilation,
)
from dynnav.researcher.orchestrator import ExperimentService
from dynnav.researcher.protocols import compile_research_request


def create_app(service: ExperimentService | None = None) -> FastAPI:
    experiment_service = service or ExperimentService()
    app = FastAPI(
        title="DynNav Researcher API",
        version="0.1.0",
        description="Evidence-bound experiment protocols, execution, statistics, and reproducibility artifacts.",
    )
    origins = os.environ.get("DYNNAV_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.experiment_service = experiment_service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "dynnav-researcher", "evidence_policy": "executed-artifacts-only"}

    @app.get("/v1/capabilities")
    def capabilities() -> dict:
        return {
            "planners": ["shortest", "risk_aware", "recoverability_aware", "proposed"],
            "scenario_families": ["seeded_random_grid"],
            "metrics": [
                "mission_success_rate",
                "irreversible_failure_rate",
                "path_length",
                "planning_time_ms",
                "cumulative_risk",
                "cumulative_irreversibility",
                "minimum_escape_options",
            ],
            "analysis": ["descriptive_statistics", "bootstrap_mean_interval", "paired_effect"],
            "unsupported_evidence": ["ROS 2", "Gazebo", "physical robot", "formal safety guarantee"],
        }

    @app.get("/v1/system")
    def system_context() -> dict:
        return experiment_service.system_context()

    @app.post("/v1/protocols", response_model=ProtocolCompilation)
    def compile_protocol(payload: NaturalLanguageProtocolRequest) -> ProtocolCompilation:
        try:
            return compile_research_request(payload.request)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    @app.post("/v1/experiments", response_model=ExperimentStatus, status_code=status.HTTP_201_CREATED)
    def create_experiment(specification: ExperimentSpecification) -> ExperimentStatus:
        return experiment_service.register(specification)

    @app.post(
        "/v1/experiments/{experiment_id}/run", response_model=ExperimentStatus, status_code=status.HTTP_202_ACCEPTED
    )
    def run_experiment(experiment_id: str) -> ExperimentStatus:
        try:
            return experiment_service.start(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.post("/v1/experiments/{experiment_id}/cancel", response_model=ExperimentStatus)
    def cancel_experiment(experiment_id: str) -> ExperimentStatus:
        try:
            return experiment_service.cancel(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @app.get("/v1/experiments/{experiment_id}", response_model=ExperimentStatus)
    def get_experiment(experiment_id: str, after_sequence: int | None = None) -> ExperimentStatus:
        try:
            return experiment_service.get_status(experiment_id, after_sequence=after_sequence)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @app.get("/v1/experiments/{experiment_id}/results", response_model=ExperimentStatus)
    def get_results(experiment_id: str) -> ExperimentStatus:
        try:
            result = experiment_service.get_status(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        if result.evidence_status not in {"completed", "partial"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="results are unavailable until executed evidence is completed or partial",
            )
        return result

    @app.get("/v1/experiments/{experiment_id}/events")
    async def experiment_events(experiment_id: str, request: Request) -> StreamingResponse:
        try:
            experiment_service.get_status(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        async def stream() -> AsyncIterator[str]:
            last_sequence = 0
            while True:
                if await request.is_disconnected():
                    return
                snapshot = experiment_service.get_status(experiment_id, after_sequence=last_sequence)
                for event in snapshot.events:
                    last_sequence = event.sequence
                    payload = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
                    yield f"id: {event.sequence}\nevent: {event.event}\ndata: {payload}\n\n"
                if snapshot.evidence_status in {"completed", "partial", "failed", "cancelled"}:
                    return
                await asyncio.sleep(0.35)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/v1/experiments/{experiment_id}/artifacts/{filename}")
    def download_artifact(experiment_id: str, filename: str) -> FileResponse:
        try:
            path = experiment_service.get_artifact_path(experiment_id, filename)
            artifact = next(
                item for item in experiment_service.get_status(experiment_id).artifacts if item.filename == filename
            )
        except (KeyError, FileNotFoundError, StopIteration) as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found") from exc
        return FileResponse(path, media_type=artifact.media_type, filename=filename)

    return app


app = create_app()
