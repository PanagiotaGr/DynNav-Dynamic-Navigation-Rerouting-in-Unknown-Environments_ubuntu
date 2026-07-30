"""Incremental D* Lite with explicit risk and recoverability transition costs.

Obstacle changes can alter the global recoverability field. For correctness, the
planner recomputes that field and refreshes affected value-function vertices while
retaining the D* Lite search state and moving-start key correction.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

from dynnav.planners.grid_map import GridCell, GridMap
from dynnav.planners.recoverability_astar import PlannerMode, RecoverabilityAStarConfig
from dynnav.recoverability import RecoverabilityState, recoverability_map

INF = float("inf")
Key = tuple[float, float]


@dataclass(frozen=True)
class DStarLiteResult:
    path: list[GridCell]
    success: bool
    cost: float
    expansions: int
    planning_time_ms: float
    cumulative_risk: float
    cumulative_irreversibility: float
    minimum_escape_options: int


class RecoverabilityDStarLite:
    """D* Lite over a :class:`GridMap` using the canonical DynNav objective."""

    def __init__(
        self,
        grid: GridMap,
        start: GridCell,
        goal: GridCell,
        *,
        safe_cells: set[GridCell] | None = None,
        mode: PlannerMode = PlannerMode.PROPOSED,
        config: RecoverabilityAStarConfig | None = None,
    ) -> None:
        grid.validate()
        if not grid.in_bounds(start) or not grid.in_bounds(goal):
            raise ValueError("start and goal must be inside the grid")
        self.grid = grid
        self.start = start
        self.goal = goal
        self.safe_cells = set(safe_cells or {start})
        self.mode = mode
        self.config = config or RecoverabilityAStarConfig()
        self.config.validate()
        self.states = recoverability_map(grid, self.safe_cells, self.config.recoverability_weights)
        self.g: dict[GridCell, float] = {}
        self.rhs: dict[GridCell, float] = {goal: 0.0}
        self.km = 0.0
        self._heap: list[tuple[float, float, GridCell]] = []
        self._active: dict[GridCell, Key] = {}
        self.total_expansions = 0
        self.replan_count = 0
        self._push(goal)

    @staticmethod
    def _heuristic(a: GridCell, b: GridCell) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _g(self, node: GridCell) -> float:
        return self.g.get(node, INF)

    def _rhs(self, node: GridCell) -> float:
        return self.rhs.get(node, INF)

    def _key(self, node: GridCell) -> Key:
        value = min(self._g(node), self._rhs(node))
        return value + self.config.step_cost * self._heuristic(self.start, node) + self.km, value

    def _mode_weights(self) -> tuple[float, float]:
        if self.mode is PlannerMode.SHORTEST:
            return 0.0, 0.0
        if self.mode is PlannerMode.RISK_AWARE:
            return self.config.risk_weight, 0.0
        if self.mode is PlannerMode.RECOVERABILITY_AWARE:
            return 0.0, self.config.irreversibility_weight
        return self.config.risk_weight, self.config.irreversibility_weight

    def _cost(self, destination: GridCell) -> float:
        if not self.grid.passable(destination):
            return INF
        risk_weight, irr_weight = self._mode_weights()
        return (
            self.config.step_cost
            + risk_weight * self.grid.cell_risk(destination)
            + irr_weight * self.states[destination].irreversibility
        )

    def _push(self, node: GridCell) -> None:
        key = self._key(node)
        self._active[node] = key
        heapq.heappush(self._heap, (key[0], key[1], node))

    def _remove(self, node: GridCell) -> None:
        self._active.pop(node, None)

    def _top_key(self) -> Key:
        while self._heap:
            k1, k2, node = self._heap[0]
            if self._active.get(node) == (k1, k2):
                return k1, k2
            heapq.heappop(self._heap)
        return INF, INF

    def _pop(self) -> tuple[GridCell, Key] | None:
        while self._heap:
            k1, k2, node = heapq.heappop(self._heap)
            if self._active.get(node) == (k1, k2):
                del self._active[node]
                return node, (k1, k2)
        return None

    def _update_vertex(self, node: GridCell) -> None:
        if not self.grid.in_bounds(node):
            return
        if node != self.goal:
            self.rhs[node] = min(
                (self._cost(successor) + self._g(successor) for successor in self.grid.neighbors4(node)),
                default=INF,
            )
        self._remove(node)
        if self._g(node) != self._rhs(node):
            self._push(node)

    def _compute(self) -> int:
        expanded = 0
        limit = max(1, self.grid.width * self.grid.height * 100)
        while self._top_key() < self._key(self.start) or self._rhs(self.start) != self._g(self.start):
            item = self._pop()
            if item is None:
                break
            node, old_key = item
            new_key = self._key(node)
            if old_key < new_key:
                self._push(node)
            elif self._g(node) > self._rhs(node):
                self.g[node] = self._rhs(node)
                for predecessor in self._adjacent(node):
                    self._update_vertex(predecessor)
            else:
                self.g[node] = INF
                self._update_vertex(node)
                for predecessor in self._adjacent(node):
                    self._update_vertex(predecessor)
            expanded += 1
            self.total_expansions += 1
            if expanded > limit:
                raise RuntimeError("recoverability D* Lite did not converge")
        return expanded

    def _adjacent(self, node: GridCell) -> list[GridCell]:
        x, y = node
        return [
            candidate
            for candidate in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if self.grid.in_bounds(candidate)
        ]

    def _extract(self) -> list[GridCell] | None:
        if self._g(self.start) == INF:
            return None
        path = [self.start]
        current = self.start
        visited = {current}
        for _ in range(self.grid.width * self.grid.height):
            if current == self.goal:
                return path
            neighbors = self.grid.neighbors4(current)
            if not neighbors:
                return None
            nxt = min(neighbors, key=lambda node: (self._cost(node) + self._g(node), node))
            if self._cost(nxt) + self._g(nxt) == INF or nxt in visited:
                return None
            path.append(nxt)
            visited.add(nxt)
            current = nxt
        return None

    def _result(self, path: list[GridCell] | None, expansions: int, elapsed_ms: float) -> DStarLiteResult:
        if not path:
            return DStarLiteResult([], False, INF, expansions, elapsed_ms, INF, INF, 0)
        costs = [self._cost(cell) for cell in path[1:]]
        return DStarLiteResult(
            path=path,
            success=True,
            cost=sum(costs),
            expansions=expansions,
            planning_time_ms=elapsed_ms,
            cumulative_risk=sum(self.grid.cell_risk(cell) for cell in path[1:]),
            cumulative_irreversibility=sum(self.states[cell].irreversibility for cell in path[1:]),
            minimum_escape_options=min(self.states[cell].escape_options for cell in path),
        )

    def plan(self) -> DStarLiteResult:
        t0 = time.perf_counter()
        if not self.grid.passable(self.start) or not self.grid.passable(self.goal):
            return self._result(None, 0, 0.0)
        expansions = self._compute()
        return self._result(self._extract(), expansions, (time.perf_counter() - t0) * 1000.0)

    def update_obstacles(self, updates: dict[GridCell, bool]) -> None:
        """Apply occupancy changes and refresh costs without discarding D* state.

        Recoverability is a global structural field, so all free vertices are
        refreshed after topology changes. This preserves correctness while still
        reusing ``g``, ``rhs``, the queue, and moving-start correction.
        """
        obstacles = set(self.grid.obstacles)
        for cell, blocked in updates.items():
            if not self.grid.in_bounds(cell):
                raise ValueError(f"occupancy update outside grid: {cell}")
            obstacles.add(cell) if blocked else obstacles.discard(cell)
        self.grid = GridMap.from_obstacles(
            self.grid.width,
            self.grid.height,
            obstacles,
            risk=self.grid.risk,
            uncertainty=self.grid.uncertainty,
        )
        self.states = recoverability_map(self.grid, self.safe_cells, self.config.recoverability_weights)
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                self._update_vertex((x, y))

    def replan(self, *, new_start: GridCell | None = None) -> DStarLiteResult:
        if new_start is not None:
            if not self.grid.in_bounds(new_start):
                raise ValueError("new start must be inside the grid")
            if new_start != self.start:
                self.km += self.config.step_cost * self._heuristic(self.start, new_start)
                self.start = new_start
        self.replan_count += 1
        return self.plan()
