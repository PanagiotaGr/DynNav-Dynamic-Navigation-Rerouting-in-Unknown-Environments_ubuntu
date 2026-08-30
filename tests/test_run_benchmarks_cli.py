from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts" / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("dynnav_run_benchmarks", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_script_respects_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "custom-output"
    planner_outputs: list[Path] = []

    monkeypatch.setattr(module, "PLANNERS", {"shortest": (), "risk": ()})
    monkeypatch.setattr(module, "load_config", lambda _: {"seed": 7})

    def fake_run_pipeline(config: dict[str, object], *, smoke: bool) -> dict[str, object]:
        planner_outputs.append(Path(str(config["output_dir"])))
        return {"planner": config["planner"], "success": smoke}

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    module.main(["--config", "unused.yaml", "--out-dir", str(output), "--smoke"])

    assert planner_outputs == [output / "shortest", output / "risk"]
    assert (output / "metrics" / "benchmark_summary.csv").is_file()
    assert (output / "reports" / "benchmark_report.md").is_file()
