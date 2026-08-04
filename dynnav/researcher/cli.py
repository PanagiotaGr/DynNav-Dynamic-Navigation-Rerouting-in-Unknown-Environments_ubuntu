"""Command-line entry point for reproducible researcher experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dynnav.researcher.models import ExperimentSpecification
from dynnav.researcher.orchestrator import ExperimentService


def _load_specification(path: Path) -> ExperimentSpecification:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return ExperimentSpecification.model_validate(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dynnav.researcher.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run a typed experiment configuration synchronously")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--artifact-root", default=Path("artifacts/researcher"), type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        service = ExperimentService(args.artifact_root)
        try:
            status = service.run_sync(_load_specification(args.config))
        finally:
            service.close()
        print(status.model_dump_json(indent=2))
        return 0 if status.evidence_status in {"completed", "partial"} else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
