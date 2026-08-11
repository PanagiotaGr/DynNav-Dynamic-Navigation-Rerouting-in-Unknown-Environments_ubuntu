"""Benchmark full recoverability-aware A* replanning against incremental D* Lite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from dynnav.planners.grid_map import GridCell, GridMap
from dynnav.planners.online_recoverability import ObstacleUpdate
from dynnav.planners.recoverability_astar import PlannerMode, RecoverabilityAStarConfig, recoverability_astar
from dynnav.planners.recoverability_dstar_lite import RecoverabilityDStarLite


@dataclass(frozen=True)
class RepairMeasurement:
    algorithm: str
    step: int
    success: bool
    path_cost: float
    path_length: int
    planning_time_ms: float
    nodes_expanded: int
    cumulative_risk: float
    cumulative_irreversibility: float
    minimum_escape_options: int


def _apply(grid: GridMap, updates: Sequence[ObstacleUpdate]) -> GridMap:
    obstacles = set(grid.obstacles)
    for update in updates:
        if not grid.in_bounds(update.cell):
            raise ValueError(f"occupancy update outside grid: {update.cell}")
        obstacles.add(update.cell) if update.blocked else obstacles.discard(update.cell)
    return GridMap.from_obstacles(
        grid.width,
        grid.height,
        obstacles,
        risk=grid.risk,
        uncertainty=grid.uncertainty,
    )


def compare_replanning_algorithms(
    grid: GridMap,
    start: GridCell,
    goal: GridCell,
    update_schedule: Mapping[int, Sequence[ObstacleUpdate]],
    *,
    mode: PlannerMode = PlannerMode.PROPOSED,
    config: RecoverabilityAStarConfig | None = None,
    safe_cells: set[GridCell] | None = None,
) -> list[RepairMeasurement]:
    """Run identical map updates through full A* and incremental D* Lite.

    The robot position is advanced one cell using the D* Lite route after each
    repair. Both algorithms are evaluated from exactly that same position and map.
    """
    config = config or RecoverabilityAStarConfig()
    safe_cells = set(safe_cells or {start})
    current_grid = grid
    position = start
    dstar = RecoverabilityDStarLite(
        current_grid,
        start,
        goal,
        safe_cells=safe_cells,
        mode=mode,
        config=config,
    )
    measurements: list[RepairMeasurement] = []
    max_step = max(update_schedule, default=0)

    for step in range(max_step + 1):
        updates = tuple(update_schedule.get(step, ()))
        if updates:
            current_grid = _apply(current_grid, updates)
            dstar.update_obstacles({update.cell: update.blocked for update in updates})

        full = recoverability_astar(
            current_grid,
            position,
            goal,
            safe_cells=safe_cells,
            mode=mode,
            config=config,
        )
        incremental = dstar.plan() if step == 0 else dstar.replan(new_start=position)

        measurements.extend(
            [
                RepairMeasurement(
                    "full_astar",
                    step,
                    full.success,
                    full.cost,
                    full.geometric_length,
                    full.planning_time_ms,
                    full.nodes_expanded,
                    full.cumulative_risk,
                    full.cumulative_irreversibility,
                    full.minimum_escape_options,
                ),
                RepairMeasurement(
                    "incremental_dstar_lite",
                    step,
                    incremental.success,
                    incremental.cost,
                    max(0, len(incremental.path) - 1),
                    incremental.planning_time_ms,
                    incremental.expansions,
                    incremental.cumulative_risk,
                    incremental.cumulative_irreversibility,
                    incremental.minimum_escape_options,
                ),
            ]
        )

        if not incremental.success or len(incremental.path) < 2:
            break
        position = incremental.path[1]
        if position == goal:
            break

    return measurements


def measurements_as_dicts(measurements: Sequence[RepairMeasurement]) -> list[dict[str, object]]:
    """Return serialization-ready benchmark rows."""
    return [asdict(row) for row in measurements]
