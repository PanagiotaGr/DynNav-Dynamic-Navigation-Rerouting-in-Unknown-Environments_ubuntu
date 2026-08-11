"""Typed occupancy-grid primitives shared by the canonical DynNav package."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Pose:
    """Discrete grid pose using ``(x, y)`` indexing."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Planned route with scalar risk and recoverability diagnostics."""

    poses: tuple[Pose, ...]
    cost: float
    risk: float
    recoverability: float

    @property
    def length(self) -> int:
        """Return the number of poses in the trajectory."""

        return len(self.poses)


@dataclass(slots=True)
class GridMap:
    """Occupancy-probability grid used by the dependency-light planners.

    Values are clipped to ``[0, 1]`` and indexed as ``occupancy[y, x]``.
    """

    occupancy: np.ndarray
    resolution: float = 1.0

    def __post_init__(self) -> None:
        if self.occupancy.ndim != 2:
            raise ValueError("occupancy must be a 2-D array")
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("resolution must be finite and positive")
        self.occupancy = np.clip(self.occupancy.astype(float), 0.0, 1.0)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the grid shape as ``(height, width)``."""

        return self.occupancy.shape

    def in_bounds(self, pose: Pose) -> bool:
        """Return whether ``pose`` lies inside the grid."""

        height, width = self.shape
        return 0 <= pose.y < height and 0 <= pose.x < width

    def probability(self, pose: Pose) -> float:
        """Return obstacle probability, treating out-of-bounds as occupied."""

        if not self.in_bounds(pose):
            return 1.0
        return float(self.occupancy[pose.y, pose.x])

    def neighbors4(self, pose: Pose) -> Iterable[Pose]:
        """Yield in-bounds four-connected neighbours."""

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            candidate = Pose(pose.x + dx, pose.y + dy)
            if self.in_bounds(candidate):
                yield candidate
