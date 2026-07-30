"""Reproducible multi-seed evaluation for recoverability-aware navigation."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from dynnav.experiments.statistics import bootstrap_mean_interval, paired_effect, summarize


@dataclass(frozen=True)
class TrialRecord:
    seed: int
    method: str
    success: bool
    irreversible_failure: bool
    path_length: float
    planning_time_ms: float
    nodes_expanded: float
    cumulative_risk: float
    cumulative_irreversibility: float
    minimum_escape_options: float


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = tuple(range(30))
    confidence: float = 0.95
    bootstrap_resamples: int = 2000

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")


TrialRunner = Callable[[int, str, Mapping[str, float]], TrialRecord]


def run_multiseed(
    runner: TrialRunner,
    methods: Sequence[str],
    *,
    parameters: Mapping[str, float] | None = None,
    config: EvaluationConfig | None = None,
) -> list[TrialRecord]:
    cfg = config or EvaluationConfig()
    cfg.validate()
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a non-empty unique sequence")
    params = dict(parameters or {})
    records: list[TrialRecord] = []
    for seed in cfg.seeds:
        random.seed(seed)
        for method in methods:
            record = runner(seed, method, params)
            if record.seed != seed or record.method != method:
                raise ValueError("runner returned mismatched seed or method")
            records.append(record)
    return records


def aggregate(records: Iterable[TrialRecord], *, config: EvaluationConfig | None = None) -> dict[str, dict]:
    cfg = config or EvaluationConfig()
    cfg.validate()
    grouped: dict[str, list[TrialRecord]] = {}
    for record in records:
        grouped.setdefault(record.method, []).append(record)
    output: dict[str, dict] = {}
    metrics = (
        "path_length", "planning_time_ms", "nodes_expanded", "cumulative_risk",
        "cumulative_irreversibility", "minimum_escape_options",
    )
    for method, rows in sorted(grouped.items()):
        summary: dict[str, object] = {
            "trials": len(rows),
            "success_rate": sum(row.success for row in rows) / len(rows),
            "irreversible_failure_rate": sum(row.irreversible_failure for row in rows) / len(rows),
        }
        for metric in metrics:
            values = [float(getattr(row, metric)) for row in rows]
            interval = bootstrap_mean_interval(
                values, confidence=cfg.confidence,
                resamples=cfg.bootstrap_resamples,
                seed=sum(row.seed for row in rows) + len(metric),
            )
            summary[metric] = {"summary": summarize(values), "mean_interval": asdict(interval)}
        output[method] = summary
    return output


def paired_comparisons(
    records: Iterable[TrialRecord], baseline: str, proposed: str,
    *, metric: str = "planning_time_ms", config: EvaluationConfig | None = None,
) -> dict:
    cfg = config or EvaluationConfig()
    cfg.validate()
    by_key = {(row.seed, row.method): row for row in records}
    common = sorted(seed for seed, method in by_key if method == baseline and (seed, proposed) in by_key)
    if not common:
        raise ValueError("no paired seeds available")
    left = [float(getattr(by_key[(seed, baseline)], metric)) for seed in common]
    right = [float(getattr(by_key[(seed, proposed)], metric)) for seed in common]
    return asdict(paired_effect(
        left, right, confidence=cfg.confidence,
        resamples=cfg.bootstrap_resamples, seed=sum(common) + len(metric),
    ))


def sensitivity_grid(
    runner: TrialRunner, method: str, parameter: str, values: Sequence[float],
    *, config: EvaluationConfig | None = None,
) -> dict[str, dict]:
    if not values:
        raise ValueError("sensitivity values cannot be empty")
    result: dict[str, dict] = {}
    for value in values:
        records = run_multiseed(runner, [method], parameters={parameter: float(value)}, config=config)
        result[str(value)] = aggregate(records, config=config)[method]
    return result


def write_artifacts(records: Sequence[TrialRecord], summary: Mapping, output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else [field.name for field in TrialRecord.__dataclass_fields__.values()])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    with (target / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
