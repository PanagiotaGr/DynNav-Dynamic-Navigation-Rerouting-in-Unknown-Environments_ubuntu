from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynnav.experiments.reproducible_pipeline import (
    PipelineConfig,
    canonical_config,
    experiment_id,
    load_config,
    run_pipeline,
)


def test_experiment_id_is_stable_and_config_sensitive() -> None:
    first = PipelineConfig(seeds=(1, 2), methods=("shortest", "proposed"))
    same = PipelineConfig(seeds=(1, 2), methods=("shortest", "proposed"))
    changed = PipelineConfig(seeds=(1, 3), methods=("shortest", "proposed"))

    assert experiment_id(first) == experiment_id(same)
    assert experiment_id(first) != experiment_id(changed)


def test_load_config_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "experiment.json"
    source.write_text(
        json.dumps(
            {
                "name": "smoke",
                "seeds": [0, 1],
                "methods": ["shortest", "proposed"],
                "confidence": 0.9,
                "bootstrap_resamples": 100,
                "scenario": {"width": 8, "height": 8},
                "parameters": {"risk_weight": 2.0},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(source)

    assert canonical_config(config)["seeds"] == [0, 1]
    assert config.methods == ("shortest", "proposed")
    assert config.parameters == {"risk_weight": 2.0}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (PipelineConfig(name="../escape"), "name must"),
        (
            PipelineConfig(methods=("shortest", "not-a-planner")),
            "unsupported planner methods",
        ),
        (
            PipelineConfig(parameters={"unknown_weight": 1.0}),
            "unsupported planner parameters",
        ),
        (
            PipelineConfig(parameters={"risk_weight": float("nan")}),
            "must be finite",
        ),
        (
            PipelineConfig(parameters={"risk_weight": -1.0}),
            "must be non-negative",
        ),
        (
            PipelineConfig(scenario={"width": 2}),
            "scenario dimensions",
        ),
    ],
)
def test_config_validation_rejects_invalid_research_inputs(
    config: PipelineConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_load_config_requires_object_sections(tmp_path: Path) -> None:
    source = tmp_path / "experiment.json"
    source.write_text(
        json.dumps({"scenario": [], "parameters": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scenario must be a JSON object"):
        load_config(source)


def test_run_pipeline_writes_reproducibility_artifacts(tmp_path: Path) -> None:
    config = PipelineConfig(
        name="smoke",
        seeds=(0,),
        methods=("shortest", "proposed"),
        bootstrap_resamples=100,
        scenario={"width": 8, "height": 8, "obstacle_probability": 0.05},
        parameters={"risk_weight": 2.0, "irreversibility_weight": 2.0},
    )

    output_dir = run_pipeline(config, tmp_path)

    assert output_dir.name == experiment_id(config)
    assert (output_dir / "trials.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "config.json").exists()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["experiment_id"] == experiment_id(config)
    assert metadata["record_count"] == 2
    assert metadata["config"]["seeds"] == [0]


def test_run_pipeline_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    config = PipelineConfig(
        name="smoke",
        seeds=(0,),
        methods=("shortest",),
        bootstrap_resamples=100,
        scenario={"width": 8, "height": 8},
    )
    output_dir = tmp_path / experiment_id(config)
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_pipeline(config, tmp_path)

    assert (output_dir / "sentinel.txt").read_text(encoding="utf-8") == "preserve"
