"""Recoverability analysis for grid navigation.

The metrics in this module are deliberately independent from occupancy risk.
They quantify structural escape and return options in the currently known free
space, so planners can distinguish a low-risk corridor from a state that is
actually easy to leave after route invalidation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from dynnav.planners.grid_map import GridCell, GridMap


@dataclass(frozen=True)
class RecoverabilityWeights:
    """Weights for the normalized irreversibility components."""

    escape_deficit: float = 1.0
    bottleneck_exposure: float = 1.0
    return_failure: float = 1.0

    def validate(self) -> None:
        values = (self.escape_deficit, self.bottleneck_exposure, self.return_failure)
        if any(value < 0.0 for value in values):
            raise ValueError("recoverability weights must be non-negative")
        if sum(values) <= 0.0:
            raise ValueError("at least one recoverability weight must be positive")


@dataclass(frozen=True)
class RecoverabilityState:
    """Normalized structural metrics for one free grid cell."""

    escape_options: int
    escape_deficit: float
    bottleneck_exposure: float
    return_failure: float
    irreversibility: float


def _reachable_without_cell(grid: GridMap, start: GridCell, blocked: GridCell) -> set[GridCell]:
    if start == blocked or not grid.in_bounds(start) or not grid.passable(start):
        return set()

    reached = {start}
    queue: deque[GridCell] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in grid.neighbors4(current):
            if neighbor == blocked or neighbor in reached:
                continue
            reached.add(neighbor)
            queue.append(neighbor)
    return reached


def escape_option_count(grid: GridMap, cell: GridCell) -> int:
    """Count locally distinct exits that remain connected without ``cell``.

    Neighbours connected to each other after removing the current cell belong to
    one escape branch. A corridor therefore has two independent options, while a
    dead-end has one and an open junction can have three or four.
    """

    neighbors = grid.neighbors4(cell)
    unseen = set(neighbors)
    components = 0
    while unseen:
        seed = unseen.pop()
        component = _reachable_without_cell(grid, seed, cell)
        unseen.difference_update(component)
        components += 1
    return components


def bottleneck_exposure(grid: GridMap, cell: GridCell) -> float:
    """Return a normalized local bottleneck score in ``[0, 1]``."""

    degree = len(grid.neighbors4(cell))
    if degree <= 1:
        return 1.0
    if degree == 2:
        return 0.75
    if degree == 3:
        return 0.25
    return 0.0


def return_failure_probability(grid: GridMap, cell: GridCell, safe_cells: set[GridCell]) -> float:
    """Estimate inability to return to any designated safe cell.

    The estimate is deterministic for the known map: zero when at least one safe
    cell is reachable, one otherwise. Dynamic uncertainty can later replace this
    binary model without changing the planner interface.
    """

    valid_safe = {safe for safe in safe_cells if grid.in_bounds(safe) and grid.passable(safe)}
    if not valid_safe:
        return 1.0
    if cell in valid_safe:
        return 0.0

    reached = _reachable_without_cell(grid, cell, blocked=(-1, -1))
    return 0.0 if reached.intersection(valid_safe) else 1.0


def analyze_recoverability(
    grid: GridMap,
    cell: GridCell,
    safe_cells: set[GridCell],
    weights: RecoverabilityWeights | None = None,
) -> RecoverabilityState:
    """Compute a normalized and auditable irreversibility score."""

    if not grid.in_bounds(cell) or not grid.passable(cell):
        raise ValueError("recoverability can only be evaluated for a free in-bounds cell")

    weights = weights or RecoverabilityWeights()
    weights.validate()

    options = escape_option_count(grid, cell)
    escape_deficit = 1.0 / (1.0 + float(options))
    bottleneck = bottleneck_exposure(grid, cell)
    return_failure = return_failure_probability(grid, cell, safe_cells)

    total_weight = weights.escape_deficit + weights.bottleneck_exposure + weights.return_failure
    irreversibility = (
        weights.escape_deficit * escape_deficit
        + weights.bottleneck_exposure * bottleneck
        + weights.return_failure * return_failure
    ) / total_weight

    return RecoverabilityState(
        escape_options=options,
        escape_deficit=escape_deficit,
        bottleneck_exposure=bottleneck,
        return_failure=return_failure,
        irreversibility=irreversibility,
    )


def recoverability_map(
    grid: GridMap,
    safe_cells: set[GridCell],
    weights: RecoverabilityWeights | None = None,
) -> dict[GridCell, RecoverabilityState]:
    """Evaluate every known free cell in the grid."""

    return {
        (x, y): analyze_recoverability(grid, (x, y), safe_cells, weights)
        for x in range(grid.width)
        for y in range(grid.height)
        if grid.passable((x, y))
    }
