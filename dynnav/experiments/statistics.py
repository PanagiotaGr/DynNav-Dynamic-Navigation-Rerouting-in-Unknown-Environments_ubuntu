"""Dependency-light statistics for reproducible navigation experiments."""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean, median, stdev


@dataclass(frozen=True)
class IntervalEstimate:
    estimate: float
    lower: float
    upper: float
    confidence: float
    sample_size: int


@dataclass(frozen=True)
class PairedEffect:
    mean_difference: float
    standardized_effect: float
    probability_of_superiority: float
    interval: IntervalEstimate


def _values(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("at least one observation is required")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("observations must be finite")
    return result


def bootstrap_mean_interval(
    values: Iterable[float], *, confidence: float = 0.95,
    resamples: int = 2000, seed: int = 0,
) -> IntervalEstimate:
    data = _values(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if len(data) == 1:
        return IntervalEstimate(data[0], data[0], data[0], confidence, 1)
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(data, k=len(data))) for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    lo = min(resamples - 1, max(0, int(tail * resamples)))
    hi = min(resamples - 1, max(0, int((1.0 - tail) * resamples) - 1))
    return IntervalEstimate(mean(data), estimates[lo], estimates[hi], confidence, len(data))


def paired_effect(
    baseline: Sequence[float], proposed: Sequence[float], *,
    confidence: float = 0.95, resamples: int = 2000, seed: int = 0,
) -> PairedEffect:
    if len(baseline) != len(proposed) or not baseline:
        raise ValueError("paired samples must have equal non-zero length")
    left, right = _values(baseline), _values(proposed)
    differences = [candidate - reference for reference, candidate in zip(left, right, strict=False)]
    spread = stdev(differences) if len(differences) > 1 else 0.0
    standardized = mean(differences) / spread if spread > 0.0 else 0.0
    wins = sum(diff < 0.0 for diff in differences)
    ties = sum(diff == 0.0 for diff in differences)
    superiority = (wins + 0.5 * ties) / len(differences)
    return PairedEffect(
        mean_difference=mean(differences),
        standardized_effect=standardized,
        probability_of_superiority=superiority,
        interval=bootstrap_mean_interval(
            differences, confidence=confidence, resamples=resamples, seed=seed
        ),
    )


def summarize(values: Iterable[float]) -> dict[str, float]:
    data = _values(values)
    return {
        "count": float(len(data)),
        "mean": mean(data),
        "median": median(data),
        "std": stdev(data) if len(data) > 1 else 0.0,
        "minimum": min(data),
        "maximum": max(data),
    }
