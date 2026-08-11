from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/contributions/experiments.yaml"
RUNNER = ROOT / "scripts/run_contribution_suite.py"


def test_experiment_registry_covers_c01_to_c26_in_order() -> None:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    ids = [entry["contribution_id"] for entry in data["experiments"]]
    assert ids == [f"C{i:02d}" for i in range(1, 27)]
    assert len({entry["experiment_id"] for entry in data["experiments"]}) == 26


def test_suite_runs_mixed_schema_and_calibration_experiments(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--only",
            "C02,C17",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [result["status"] for result in manifest["results"]] == ["passed", "passed"]
    assert all(result["artifact_sha256"] for result in manifest["results"])

    with (tmp_path / "artifacts/c17_topological_semantic_navigation.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {row["row_type"] for row in rows} == {"grounding", "planning"}
    assert all("path_found" in row and "top1_correct" in row for row in rows)
