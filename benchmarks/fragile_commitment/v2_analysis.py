"""Predefined primary summaries for the bias-controlled V2 benchmark."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from dynnav.evaluation.scientific_metrics import paired_risk_difference, wilson_interval

PRIMARY_CONTRASTS = (("J0_shortest", "J2_recoverability"), ("J1_risk", "J3_joint"))


def _bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def analyse(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    indexed: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["condition"], row["planner"])].append(row)
        indexed[(row["condition"], int(row["seed"]))][row["planner"]] = row

    summaries: list[dict[str, object]] = []
    for (condition, planner), group in sorted(grouped.items()):
        failures = sum(not _bool(row["mission_success"]) for row in group)
        low, high = wilson_interval(failures, len(group))
        summaries.append(
            {
                "condition": condition,
                "planner": planner,
                "trials": len(group),
                "failure_rate": failures / len(group),
                "failure_ci95_low": low,
                "failure_ci95_high": high,
                "mean_path_length": sum(float(row["path_length"]) for row in group) / len(group),
                "mean_route_risk": sum(float(row["route_risk"]) for row in group) / len(group),
                "mean_min_recoverability": sum(float(row["min_recoverability"]) for row in group) / len(group),
            }
        )

    contrasts: list[dict[str, object]] = []
    for condition in sorted({key[0] for key in indexed}):
        pairs = [value for (name, _), value in indexed.items() if name == condition]
        for baseline, candidate in PRIMARY_CONTRASTS:
            baseline_failures = [not _bool(pair[baseline]["mission_success"]) for pair in pairs]
            candidate_failures = [not _bool(pair[candidate]["mission_success"]) for pair in pairs]
            difference, baseline_only, candidate_only = paired_risk_difference(
                baseline_failures, candidate_failures
            )
            contrasts.append(
                {
                    "condition": condition,
                    "baseline": baseline,
                    "candidate": candidate,
                    "pairs": len(pairs),
                    "candidate_minus_baseline_failure_risk": difference,
                    "baseline_only_failures": baseline_only,
                    "candidate_only_failures": candidate_only,
                }
            )
    return summaries, contrasts


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("V2 results are empty")
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--contrasts", type=Path, required=True)
    args = parser.parse_args()
    summaries, contrasts = analyse(_read(args.results))
    _write(args.summary, summaries)
    _write(args.contrasts, contrasts)


if __name__ == "__main__":
    main()
