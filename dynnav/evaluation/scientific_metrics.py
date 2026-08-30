"""Operational, unit-tested quantities for the DynNav core hypothesis.

These functions deliberately separate robot-visible scores from evaluation-only
ground truth.  They do not convert a heuristic into a probability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Minimal evidence needed to classify one dynamic-navigation trial."""

    event_injected: bool
    event_observed: bool
    pre_event_path_invalidated: bool
    mission_success: bool
    recovery_assessed: bool
    recovery_feasible: bool | None
    collision: bool = False
    emergency_stop: bool = False
    timeout: bool = False

    @property
    def valid_invalidation(self) -> bool:
        return self.event_injected and self.event_observed and self.pre_event_path_invalidated

    @property
    def irreversible_failure(self) -> bool | None:
        """Operational post-invalidation recovery-infeasible failure label.

        ``None`` means the primary quantity is not assessable; it must not be
        silently converted to success or failure.
        """

        if not self.valid_invalidation or not self.recovery_assessed:
            return None
        if self.recovery_feasible is None:
            raise ValueError("assessed recovery requires a feasibility result")
        mission_failed = not self.mission_success
        return mission_failed and not self.recovery_feasible

    @property
    def failure_reason(self) -> str:
        if not self.event_injected:
            return "invalid_event_not_injected"
        if not self.event_observed:
            return "invalid_event_not_observed"
        if not self.pre_event_path_invalidated:
            return "valid_negative_control"
        if self.collision:
            return "collision"
        if self.emergency_stop:
            return "emergency_stop"
        if self.timeout:
            return "timeout"
        if self.mission_success:
            return "succeeded"
        if not self.recovery_assessed:
            return "failure_unassessed_recovery"
        return "recovery_infeasible" if self.recovery_feasible is False else "mission_failed_recovery_feasible"


def executed_path_length(
    xy: Sequence[tuple[float, float]], *, reset_jump_threshold_m: float | None = None
) -> float:
    """Polyline length, optionally excluding simulator reset/teleport jumps."""

    if reset_jump_threshold_m is not None and (
        not math.isfinite(reset_jump_threshold_m) or reset_jump_threshold_m <= 0.0
    ):
        raise ValueError("reset jump threshold must be finite and positive")
    total = 0.0
    for left, right in zip(xy, xy[1:], strict=False):
        values = (*left, *right)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("trajectory coordinates must be finite")
        distance = math.dist(left, right)
        if reset_jump_threshold_m is None or distance <= reset_jump_threshold_m:
            total += distance
    return total


def relative_overhead(candidate: float, reference: float) -> float:
    """Return `(candidate-reference)/reference`; zero reference is undefined."""

    if not all(math.isfinite(value) for value in (candidate, reference)):
        raise ValueError("costs must be finite")
    if candidate < 0.0 or reference <= 0.0:
        raise ValueError("candidate must be non-negative and reference positive")
    return (candidate - reference) / reference


def time_integral(times_s: Sequence[float], values: Sequence[float]) -> float:
    """Trapezoidal integral for cumulative risk or option preservation."""

    if len(times_s) != len(values) or not times_s:
        raise ValueError("times and values must have equal non-zero length")
    if not all(math.isfinite(value) for value in (*times_s, *values)):
        raise ValueError("times and values must be finite")
    if any(right <= left for left, right in zip(times_s, times_s[1:], strict=False)):
        raise ValueError("timestamps must be strictly increasing")
    return float(np.trapezoid(np.asarray(values, dtype=float), np.asarray(times_s, dtype=float)))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - margin)
    high = 1.0 if successes == trials else min(1.0, centre + margin)
    return low, high


def paired_risk_difference(
    baseline_failures: Sequence[bool], candidate_failures: Sequence[bool]
) -> tuple[float, int, int]:
    """Candidate-minus-baseline failure risk and discordant counts."""

    if len(baseline_failures) != len(candidate_failures) or not baseline_failures:
        raise ValueError("paired outcomes must have equal non-zero length")
    baseline_only = 0
    candidate_only = 0
    for left, right in zip(baseline_failures, candidate_failures, strict=True):
        baseline_only += int(left and not right)
        candidate_only += int(right and not left)
    difference = (candidate_only - baseline_only) / len(baseline_failures)
    return difference, baseline_only, candidate_only
