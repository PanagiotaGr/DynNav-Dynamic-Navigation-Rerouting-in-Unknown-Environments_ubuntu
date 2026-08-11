#!/usr/bin/env python3
"""Run one canonical, dependency-aware smoke experiment for each C01-C26 module."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs/contributions/experiments.yaml"


def load_experiments(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiments = data.get("experiments", []) if isinstance(data, dict) else []
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiment registry must contain a non-empty experiments list")
    ids = [str(item["contribution_id"]) for item in experiments]
    expected = [f"C{i:02d}" for i in range(1, 27)]
    if ids != expected:
        raise ValueError(f"experiment registry must be ordered exactly C01-C26; got {ids}")
    return experiments


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def missing_dependencies(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def run_experiment(
    experiment: dict[str, Any],
    output_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    cid = str(experiment["contribution_id"])
    missing = missing_dependencies([str(name) for name in experiment["optional_dependencies"]])
    artifact = output_dir / "artifacts" / str(experiment["artifact"])
    command = [
        sys.executable,
        str(ROOT / experiment["runner"]),
        *[str(value) for value in experiment["smoke_arguments"]],
        str(experiment["output_argument"]),
        str(artifact),
    ]
    result: dict[str, Any] = {
        "contribution_id": cid,
        "experiment_id": experiment["experiment_id"],
        "status": "pending",
        "command": command,
        "artifact": str(artifact.relative_to(output_dir)),
        "evidence_level": experiment["evidence_level"],
        "claim_boundary": experiment["limitation"],
        "missing_dependencies": missing,
    }
    if missing:
        result["status"] = "skipped_optional_dependency"
        result["duration_seconds"] = 0.0
        return result

    artifact.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            status="timed_out",
            duration_seconds=round(time.perf_counter() - started, 3),
            stdout=(exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        )
        return result

    result.update(
        status="passed" if process.returncode == 0 and artifact.is_file() else "failed",
        returncode=process.returncode,
        duration_seconds=round(time.perf_counter() - started, 3),
        stdout=process.stdout[-4000:],
        stderr=process.stderr[-4000:],
    )
    if artifact.is_file():
        result["artifact_sha256"] = sha256_file(artifact)
        result["artifact_rows"] = csv_rows(artifact)
        result["artifact_bytes"] = artifact.stat().st_size
    elif process.returncode == 0:
        result["status"] = "missing_artifact"
    return result


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    lines = [
        "# DynNav C01–C26 smoke-suite summary",
        "",
        "> These are controlled implementation checks, not real-robot or generalization evidence.",
        "",
        f"- Registered: {len(results)}",
        *[f"- {status}: {count}" for status, count in sorted(counts.items())],
        "",
        "| ID | Experiment | Status | Rows | SHA-256 |",
        "|---|---|---|---:|---|",
    ]
    for result in results:
        digest = str(result.get("artifact_sha256", "—"))
        if digest != "—":
            digest = f"`{digest[:12]}…`"
        lines.append(
            f"| {result['contribution_id']} | `{result['experiment_id']}` | "
            f"{result['status']} | {result.get('artifact_rows', '—')} | {digest} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=Path("results/contribution_suite"))
    parser.add_argument("--only", help="Comma-separated contribution IDs, for example C02,C17")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--fail-on-skip", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    experiments = load_experiments(args.registry.resolve())
    if args.only:
        selected = {item.strip().upper() for item in args.only.split(",") if item.strip()}
        experiments = [item for item in experiments if item["contribution_id"] in selected]
        missing = selected - {item["contribution_id"] for item in experiments}
        if missing:
            parser.error(f"unknown contribution IDs: {sorted(missing)}")
    if args.list:
        for item in experiments:
            dependencies = ",".join(item["optional_dependencies"]) or "none"
            print(f"{item['contribution_id']}\t{item['experiment_id']}\toptional={dependencies}")
        return 0

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_experiment(item, output_dir, args.timeout) for item in experiments]
    payload = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "claim_boundary": (
            "Controlled implementation smoke checks only; no real-robot, Gazebo, formal-proof, "
            "trained-model, or generalization claim."
        ),
        "results": results,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(output_dir / "summary.md", results)

    for result in results:
        print(f"{result['contribution_id']} {result['status']}: {result['experiment_id']}")
    failed = any(result["status"] not in {"passed", "skipped_optional_dependency"} for result in results)
    skipped = any(result["status"] == "skipped_optional_dependency" for result in results)
    return 1 if failed or (args.fail_on_skip and skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
