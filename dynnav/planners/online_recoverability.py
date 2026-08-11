"""Online replanning with explicit recoverability-aware objectives.

The controller intentionally separates environment updates, planning decisions,
and execution. Every map change triggers a fresh recoverability analysis and a
new plan from the robot's current state. This provides a deterministic reference
implementation for dynamic-route-invalidation experiments.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from dynnav.planners.grid_map import GridCell, GridMap
from dynnav.planners.recoverability_astar import (
    PlannerMode,
    RecoverabilityAStarConfig,
    RecoverabilityAStarResult,
    recoverability_astar,
)
from dynnav.recoverability import analyze_recoverability


class EpisodeStatus(str, Enum):
    SUCCESS = "success"
    NO_INITIAL_PATH = "no_initial_path"
    ROUTE_INVALIDATED = "route_invalidated"
    IRREVERSIBLE_FAILURE = "irreversible_failure"
    STEP_LIMIT = "step_limit"


@dataclass(frozen=True)
class ObstacleUpdate:
    """A deterministic occupancy update applied before planning at one step."""

    cell: GridCell
    blocked: bool = True


@dataclass(frozen=True)
class ReplanningStep:
    step: int
    position: GridCell
    applied_updates: tuple[ObstacleUpdate, ...]
    selected_path: tuple[GridCell, ...]
    planning_success: bool
    planning_time_ms: float
    nodes_expanded: int
    path_cost: float
    cumulative_risk: float
    cumulative_irreversibility: float
    escape_options: int
    irreversibility: float


@dataclass(frozen=True)
class OnlineReplanningResult:
    mode: PlannerMode
    status: EpisodeStatus
    success: bool
    irreversible_failure: bool
    trajectory: tuple[GridCell, ...]
    steps: tuple[ReplanningStep, ...]
    replanning_count: int
    total_planning_time_ms: float
    total_nodes_expanded: int
    path_length: int
    cumulative_executed_risk: float
    minimum_escape_options: int


@dataclass
class OnlineRecoverabilityPlanner:
    """Execute one grid action at a time and replan after observations.

    ``update_schedule`` maps an execution step to occupancy changes. Updates at
    step zero are applied before the initial plan. At later steps they are
    applied after the robot has reached the current state and before its next
    action. The implementation is deliberately stateless with respect to A*;
    it is a scientific baseline for validating the objective before an
    incremental D* Lite optimisation is introduced.
    """

    grid: GridMap
    start: GridCell
    goal: GridCell
    mode: PlannerMode = PlannerMode.PROPOSED
    config: RecoverabilityAStarConfig = field(default_factory=RecoverabilityAStarConfig)
    safe_cells: set[GridCell] | None = None

    def __post_init__(self) -> None:
        self.grid.validate()
        self.config.validate()
        if not self.grid.in_bounds(self.start) or not self.grid.in_bounds(self.goal):
            raise ValueError("start and goal must be inside the grid")
        self.safe_cells = set(self.safe_cells or {self.start})

    @staticmethod
    def _apply_updates(grid: GridMap, updates: Iterable[ObstacleUpdate]) -> GridMap:
        obstacles = set(grid.obstacles)
        for update in updates:
            if not grid.in_bounds(update.cell):
                raise ValueError(f"occupancy update outside grid: {update.cell}")
            if update.blocked:
                obstacles.add(update.cell)
            else:
                obstacles.discard(update.cell)
        return GridMap.from_obstacles(
            width=grid.width,
            height=grid.height,
            obstacles=obstacles,
            risk=grid.risk,
            uncertainty=grid.uncertainty,
        )

    @staticmethod
    def _can_reach_safe_region(grid: GridMap, start: GridCell, safe_cells: set[GridCell]) -> bool:
        if not grid.in_bounds(start) or not grid.passable(start):
            return False
        valid_safe = {cell for cell in safe_cells if grid.in_bounds(cell) and grid.passable(cell)}
        if not valid_safe:
            return False
        queue: deque[GridCell] = deque([start])
        reached = {start}
        while queue:
            current = queue.popleft()
            if current in valid_safe:
                return True
            for neighbor in grid.neighbors4(current):
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        return False

    def _plan(self, grid: GridMap, position: GridCell) -> RecoverabilityAStarResult:
        return recoverability_astar(
            grid,
            position,
            self.goal,
            safe_cells=set(self.safe_cells or {self.start}),
            mode=self.mode,
            config=self.config,
        )

    def run(
        self,
        update_schedule: Mapping[int, Sequence[ObstacleUpdate]] | None = None,
        *,
        max_steps: int | None = None,
    ) -> OnlineReplanningResult:
        schedule = {int(step): tuple(updates) for step, updates in (update_schedule or {}).items()}
        if any(step < 0 for step in schedule):
            raise ValueError("update steps must be non-negative")
        step_limit = max_steps if max_steps is not None else self.grid.width * self.grid.height * 4
        if step_limit <= 0:
            raise ValueError("max_steps must be positive")

        current_grid = self.grid
        position = self.start
        trajectory = [position]
        records: list[ReplanningStep] = []
        total_time = 0.0
        total_nodes = 0
        executed_risk = 0.0
        minimum_escape = 4
        replans = 0

        for step in range(step_limit + 1):
            updates = schedule.get(step, ())
            if updates:
                current_grid = self._apply_updates(current_grid, updates)
                if position in current_grid.obstacles:
                    return self._finish(
                        EpisodeStatus.IRREVERSIBLE_FAILURE,
                        trajectory,
                        records,
                        replans,
                        total_time,
                        total_nodes,
                        executed_risk,
                        minimum_escape,
                    )

            if position == self.goal:
                return self._finish(
                    EpisodeStatus.SUCCESS,
                    trajectory,
                    records,
                    replans,
                    total_time,
                    total_nodes,
                    executed_risk,
                    minimum_escape,
                )

            plan = self._plan(current_grid, position)
            total_time += plan.planning_time_ms
            total_nodes += plan.nodes_expanded
            if step > 0 or updates:
                replans += 1

            state = analyze_recoverability(
                current_grid,
                position,
                set(self.safe_cells or {self.start}),
                self.config.recoverability_weights,
            )
            minimum_escape = min(minimum_escape, state.escape_options)
            records.append(
                ReplanningStep(
                    step=step,
                    position=position,
                    applied_updates=updates,
                    selected_path=tuple(plan.path),
                    planning_success=plan.success,
                    planning_time_ms=plan.planning_time_ms,
                    nodes_expanded=plan.nodes_expanded,
                    path_cost=plan.cost,
                    cumulative_risk=plan.cumulative_risk,
                    cumulative_irreversibility=plan.cumulative_irreversibility,
                    escape_options=state.escape_options,
                    irreversibility=state.irreversibility,
                )
            )

            if not plan.success or len(plan.path) < 2:
                irreversible = not self._can_reach_safe_region(
                    current_grid, position, set(self.safe_cells or {self.start})
                )
                status = EpisodeStatus.IRREVERSIBLE_FAILURE if irreversible else (
                    EpisodeStatus.NO_INITIAL_PATH if step == 0 else EpisodeStatus.ROUTE_INVALIDATED
                )
                return self._finish(
                    status,
                    trajectory,
                    records,
                    replans,
                    total_time,
                    total_nodes,
                    executed_risk,
                    minimum_escape,
                )

            position = plan.path[1]
            trajectory.append(position)
            executed_risk += current_grid.cell_risk(position)

        return self._finish(
            EpisodeStatus.STEP_LIMIT,
            trajectory,
            records,
            replans,
            total_time,
            total_nodes,
            executed_risk,
            minimum_escape,
        )

    def _finish(
        self,
        status: EpisodeStatus,
        trajectory: list[GridCell],
        records: list[ReplanningStep],
        replans: int,
        total_time: float,
        total_nodes: int,
        executed_risk: float,
        minimum_escape: int,
    ) -> OnlineReplanningResult:
        success = status is EpisodeStatus.SUCCESS
        return OnlineReplanningResult(
            mode=self.mode,
            status=status,
            success=success,
            irreversible_failure=status is EpisodeStatus.IRREVERSIBLE_FAILURE,
            trajectory=tuple(trajectory),
            steps=tuple(records),
            replanning_count=replans,
            total_planning_time_ms=total_time,
            total_nodes_expanded=total_nodes,
            path_length=max(0, len(trajectory) - 1),
            cumulative_executed_risk=executed_risk,
            minimum_escape_options=0 if minimum_escape == 4 and not records else minimum_escape,
        )
