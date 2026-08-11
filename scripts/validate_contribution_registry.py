#!/usr/bin/env python3
"""Validate the canonical DynNav contribution registry without importing Streamlit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ALLOWED_STATUS = {
    "Implemented",
    "Research Prototype",
    "Experimental",
    "Synthetic Validation",
    "Dataset Validation",
    "Simulation Validation",
    "Documentation Concept",
    "Planned",
    "Missing Implementation",
    "Deprecated",
    "Duplicate",
    "ROS 2 Validation Pending",
    "Hardware Validation Required",
}
ALLOWED_EVIDENCE_LEVEL = {"synthetic", "simulation", "dataset", "hardware", "real_robot", "formal"}
EXPERIMENT_REQUIRED_FIELDS = {
    "contribution_id",
    "experiment_id",
    "hypothesis",
    "runner",
    "smoke_arguments",
    "output_argument",
    "artifact",
    "optional_dependencies",
    "baselines",
    "primary_metrics",
    "evidence_level",
    "limitation",
}


def load_registry(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("contributions"), list):
        raise ValueError("registry must contain a contributions list")
    return data


def validate(root: Path, registry_path: Path) -> dict[str, Any]:
    registry = load_registry(registry_path)
    entries = registry["contributions"]
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    slugs: set[str] = set()

    for entry in entries:
        cid = str(entry.get("id", ""))
        slug = str(entry.get("slug", ""))
        if cid in ids:
            errors.append(f"duplicate id: {cid}")
        if slug in slugs:
            errors.append(f"duplicate slug: {slug}")
        ids.add(cid)
        slugs.add(slug)
        if entry.get("status") not in ALLOWED_STATUS:
            errors.append(f"{cid}: invalid status {entry.get('status')!r}")
        directory = root / str(entry.get("directory", ""))
        if not directory.is_dir():
            errors.append(f"{cid}: missing directory {directory.relative_to(root)}")
        for target in entry.get("integrates_with", []):
            if target == cid:
                errors.append(f"{cid}: self dependency")

    expected = {f"C{i:02d}" for i in range(1, 27)}
    if ids != expected:
        errors.append(f"registry IDs differ from C01-C26: missing={sorted(expected-ids)}, extra={sorted(ids-expected)}")
    for entry in entries:
        for target in entry.get("integrates_with", []):
            if target not in ids:
                errors.append(f"{entry['id']}: unknown dependency {target}")

    experiment_count = 0
    experiment_registry = root / str(registry.get("experiment_registry", "configs/contributions/experiments.yaml"))
    if not experiment_registry.is_file():
        errors.append(f"missing experiment registry: {experiment_registry.relative_to(root)}")
    else:
        experiment_data = yaml.safe_load(experiment_registry.read_text(encoding="utf-8"))
        experiments = experiment_data.get("experiments", []) if isinstance(experiment_data, dict) else []
        if not isinstance(experiments, list):
            errors.append("experiment registry must contain an experiments list")
            experiments = []
        experiment_count = len(experiments)
        experiment_ids: set[str] = set()
        contribution_ids: set[str] = set()
        for experiment in experiments:
            if not isinstance(experiment, dict):
                errors.append("experiment entries must be mappings")
                continue
            missing_fields = sorted(EXPERIMENT_REQUIRED_FIELDS - set(experiment))
            cid = str(experiment.get("contribution_id", "unknown"))
            if missing_fields:
                errors.append(f"{cid}: missing experiment fields {missing_fields}")
            experiment_id = str(experiment.get("experiment_id", ""))
            if experiment_id in experiment_ids:
                errors.append(f"duplicate experiment id: {experiment_id}")
            if cid in contribution_ids:
                errors.append(f"{cid}: more than one canonical smoke experiment")
            experiment_ids.add(experiment_id)
            contribution_ids.add(cid)
            runner = root / str(experiment.get("runner", ""))
            if not runner.is_file():
                errors.append(f"{cid}: missing experiment runner {runner.relative_to(root)}")
            if not isinstance(experiment.get("smoke_arguments"), list):
                errors.append(f"{cid}: smoke_arguments must be a list")
            if not isinstance(experiment.get("optional_dependencies"), list):
                errors.append(f"{cid}: optional_dependencies must be a list")
            for field in ("baselines", "primary_metrics"):
                if not isinstance(experiment.get(field), list) or not experiment.get(field):
                    errors.append(f"{cid}: {field} must be a non-empty list")
            if experiment.get("evidence_level") not in ALLOWED_EVIDENCE_LEVEL:
                errors.append(f"{cid}: invalid evidence_level {experiment.get('evidence_level')!r}")
            artifact = Path(str(experiment.get("artifact", "")))
            if artifact.name != str(experiment.get("artifact", "")) or artifact.suffix != ".csv":
                errors.append(f"{cid}: artifact must be a CSV basename")
        if contribution_ids != expected:
            errors.append(
                "experiment contribution IDs differ from C01-C26: "
                f"missing={sorted(expected-contribution_ids)}, extra={sorted(contribution_ids-expected)}"
            )

    return {
        "schema_version": 1,
        "registry": str(registry_path.relative_to(root)),
        "contribution_count": len(entries),
        "experiment_count": experiment_count,
        "ids": sorted(ids),
        "errors": errors,
        "warnings": warnings,
        "status": "passed" if not errors else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registry", type=Path, default=Path("configs/contributions/registry.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = validate(root, root / args.registry)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = root / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
