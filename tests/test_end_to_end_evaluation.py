from pathlib import Path

import pytest

from dynnav.experiments.end_to_end_evaluation import DEFAULT_METHODS, make_runner, run_experiment
from dynnav.experiments.multiseed_evaluation import EvaluationConfig
from dynnav.experiments.scenario_suite import ScenarioConfig, generate_scenario


def test_scenario_generation_is_deterministic_and_solvable():
    left = generate_scenario(7)
    right = generate_scenario(7)
    assert left.grid.obstacles == right.grid.obstacles
    assert left.grid.risk == right.grid.risk
    x, y = left.start
    while x < left.goal[0]:
        x += 1
        assert left.grid.passable((x, y))
    while y < left.goal[1]:
        y += 1
        assert left.grid.passable((x, y))


def test_runner_supports_all_canonical_modes():
    runner = make_runner(ScenarioConfig(width=8, height=6))
    for method in DEFAULT_METHODS:
        record = runner(3, method, {})
        assert record.method == method
        assert record.seed == 3
        assert record.success
        assert record.path_length > 0


def test_runner_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown planner method"):
        make_runner()(0, "unknown", {})


def test_end_to_end_experiment_writes_artifacts(tmp_path: Path):
    config = EvaluationConfig(seeds=(1, 2, 3), bootstrap_resamples=100)
    result = run_experiment(
        evaluation_config=config,
        scenario_config=ScenarioConfig(width=8, height=6),
        output_dir=tmp_path,
    )
    assert len(result.records) == 3 * len(DEFAULT_METHODS)
    assert set(result.summary) == set(DEFAULT_METHODS)
    assert "cumulative_irreversibility" in result.comparisons
    assert (tmp_path / "trials.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_scenario_config_validation():
    with pytest.raises(ValueError):
        ScenarioConfig(width=3).validate()
    with pytest.raises(ValueError):
        ScenarioConfig(obstacle_probability=1.0).validate()
