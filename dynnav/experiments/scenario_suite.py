"""Deterministic synthetic scenarios for recoverability experiments."""
from __future__ import annotations

import random
from dataclasses import dataclass

from dynnav.planners.grid_map import GridCell, GridMap


@dataclass(frozen=True)
class ScenarioConfig:
    width: int = 20
    height: int = 14
    obstacle_probability: float = 0.18
    risk_probability: float = 0.25
    maximum_risk: float = 0.9

    def validate(self) -> None:
        if self.width < 4 or self.height < 4:
            raise ValueError("scenario dimensions must be at least 4")
        if not 0.0 <= self.obstacle_probability < 1.0:
            raise ValueError("obstacle_probability must be in [0, 1)")
        if not 0.0 <= self.risk_probability <= 1.0:
            raise ValueError("risk_probability must be in [0, 1]")
        if not 0.0 <= self.maximum_risk <= 1.0:
            raise ValueError("maximum_risk must be in [0, 1]")


@dataclass(frozen=True)
class NavigationScenario:
    seed: int
    grid: GridMap
    start: GridCell
    goal: GridCell
    safe_cells: frozenset[GridCell]


def generate_scenario(seed: int, config: ScenarioConfig | None = None) -> NavigationScenario:
    """Generate a reproducible map while reserving a guaranteed start-goal corridor."""
    cfg = config or ScenarioConfig()
    cfg.validate()
    rng = random.Random(seed)
    start = (0, 0)
    goal = (cfg.width - 1, cfg.height - 1)
    corridor = {(x, 0) for x in range(cfg.width)} | {
        (cfg.width - 1, y) for y in range(cfg.height)
    }
    obstacles: set[GridCell] = set()
    risk: dict[GridCell, float] = {}
    uncertainty: dict[GridCell, float] = {}
    for x in range(cfg.width):
        for y in range(cfg.height):
            cell = (x, y)
            if cell not in corridor and rng.random() < cfg.obstacle_probability:
                obstacles.add(cell)
                continue
            if cell not in (start, goal) and rng.random() < cfg.risk_probability:
                risk[cell] = rng.random() * cfg.maximum_risk
            uncertainty[cell] = rng.random()
    grid = GridMap.from_obstacles(
        cfg.width, cfg.height, obstacles=obstacles, risk=risk, uncertainty=uncertainty
    )
    return NavigationScenario(seed, grid, start, goal, frozenset({start}))
