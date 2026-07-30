"""Configuration-driven, reproducible experiment orchestration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dynnav.experiments.end_to_end_evaluation import DEFAULT_METHODS, run_experiment
from dynnav.experiments.multiseed_evaluation import EvaluationConfig
from dynnav.experiments.scenario_suite import ScenarioConfig


@dataclass(frozen=True)
class PipelineConfig:
    """Serializable configuration for one reproducible experiment run."""

    name: str = "recoverability_astar"
    seeds: tuple[int, ...] = tuple(range(30))
    methods: tuple[str, ...] = DEFAULT_METHODS
    confidence: float = 0.95
    bootstrap_resamples: int = 2000
    scenario: Mapping[str, Any] | None = None
    parameters: Mapping[str, float] | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        EvaluationConfig(
            seeds=self.seeds,
            confidence=self.confidence,
            bootstrap_resamples=self.bootstrap_resamples,
        ).validate()
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be a non-empty unique sequence")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PipelineConfig":
        return cls(
            name=str(data.get("name", "recoverability_astar")),
            seeds=tuple(int(seed) for seed in data.get("seeds", range(30))),
            methods=tuple(str(method) for method in data.get("methods", DEFAULT_METHODS)),
            confidence=float(data.get("confidence", 0.95)),
            bootstrap_resamples=int(data.get("bootstrap_resamples", 2000)),
            scenario=dict(data.get("scenario", {})),
            parameters={str(key): float(value) for key, value in data.get("parameters", {}).items()},
        )


def load_config(path: str | Path) -> PipelineConfig:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    config = PipelineConfig.from_mapping(data)
    config.validate()
    return config


def canonical_config(config: PipelineConfig) -> dict[str, Any]:
    data = asdict(config)
    data["seeds"] = list(config.seeds)
    data["methods"] = list(config.methods)
    data["scenario"] = dict(config.scenario or {})
    data["parameters"] = dict(config.parameters or {})
    return data


def experiment_id(config: PipelineConfig) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{config.name}-{digest}"


def resolve_commit_sha() -> str | None:
    for variable in ("GITHUB_SHA", "CI_COMMIT_SHA"):
        if value := os.getenv(variable):
            return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_pipeline(config: PipelineConfig, output_root: str | Path) -> Path:
    config.validate()
    run_id = experiment_id(config)
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = EvaluationConfig(
        seeds=config.seeds,
        confidence=config.confidence,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    scenario_config = ScenarioConfig(**dict(config.scenario or {}))
    result = run_experiment(
        methods=config.methods,
        evaluation_config=evaluation_config,
        scenario_config=scenario_config,
        parameters=config.parameters,
        output_dir=output_dir,
    )

    metadata = {
        "experiment_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": resolve_commit_sha(),
        "config": canonical_config(config),
        "record_count": len(result.records),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(canonical_config(config), handle, indent=2, sort_keys=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON experiment configuration")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = run_pipeline(load_config(args.config), args.output_root)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
