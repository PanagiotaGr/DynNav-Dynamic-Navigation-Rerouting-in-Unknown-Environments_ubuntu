from dataclasses import fields
from pathlib import Path

import yaml

from dynnav_dashboard.config import ScenarioConfig


def test_presentation_config_matches_scenario_schema() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "presentation.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    allowed = {field.name for field in fields(ScenarioConfig)}
    assert set(raw) <= allowed

    raw["start"] = tuple(raw["start"])
    raw["goal"] = tuple(raw["goal"])
    config = ScenarioConfig(**raw)

    assert config.start != config.goal
    assert 0 <= config.start[0] < config.grid_size
    assert 0 <= config.start[1] < config.grid_size
    assert 0 <= config.goal[0] < config.grid_size
    assert 0 <= config.goal[1] < config.grid_size
    assert config.n_dynamic_obstacles > 0
    assert config.risk_weight > 0
    assert config.max_steps > 0
