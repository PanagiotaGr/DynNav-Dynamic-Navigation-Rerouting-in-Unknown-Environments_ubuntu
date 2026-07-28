#!/usr/bin/env python3
"""Run reproducible multi-seed DynNav presentation benchmarks.

The script compares classical A* and risk-aware A* on identical generated
worlds, then evaluates both planners in the closed-loop dynamic rollout. It
writes raw per-seed CSV data, an aggregate CSV table, and a Markdown table that
can be pasted directly into a report or slide deck.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from dynnav_dashboard.config import DEFAULT_SCENARIO, ScenarioConfig
from dynnav_dashboard.simulation import (
    build_environment,
    plan_astar,
    plan_risk_aware,
    simulate_rollout,
)

RAW_FIELDS = [
    "seed",
    "planner",
    "static_success",
    "static_path_cells",
    "static_expansions",
    "static_runtime_ms",
    "static_cost",
    "static_avg_risk",
    "static_max_risk",
    "rollout_success",
    "rollout_distance",
    "rollout_replans",
    "rollout_avg_risk",
    "rollout_max_risk",
    "rollout_avg_compute_ms",
    "rollout_collisions",
]

METRICS = [
    "static_success",
    "static_path_cells",
    "static_expansions",
    "static_runtime_ms",
    "static_cost",
    "static_avg_risk",
    "static_max_risk",
    "rollout_success",
    "rollout_distance",
    "rollout_replans",
    "rollout_avg_risk",
    "rollout_max_risk",
    "rollout_avg_compute_ms",
    "rollout_collisions",
]


def parse_seeds(value: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges such as 1-5,9."""
    seeds: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid descending seed range: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return list(dict.fromkeys(seeds))


def finite(value: float) -> float:
    """Convert non-finite planner costs into an empty CSV value upstream."""
    return value if math.isfinite(value) else float("nan")


def run_seed(cfg: ScenarioConfig, seed: int, dynamic_step_every: int) -> list[dict[str, object]]:
    """Evaluate both planners on one identical environment seed."""
    env = build_environment(cfg, seed=seed)
    rows: list[dict[str, object]] = []

    for planner_name, use_risk_aware in (("A*", False), ("Risk-aware A*", True)):
        static = (
            plan_risk_aware(env, cfg.start, cfg.goal, cfg.risk_weight)
            if use_risk_aware
            else plan_astar(env, cfg.start, cfg.goal)
        )
        rollout = simulate_rollout(
            env,
            cfg,
            use_risk_aware=use_risk_aware,
            dynamic_step_every=dynamic_step_every,
        )
        rows.append(
            {
                "seed": seed,
                "planner": planner_name,
                "static_success": int(static.success),
                "static_path_cells": len(static.path),
                "static_expansions": static.expansions,
                "static_runtime_ms": static.runtime_ms,
                "static_cost": finite(static.cost),
                "static_avg_risk": static.avg_risk,
                "static_max_risk": static.max_risk,
                "rollout_success": int(rollout.reached_goal),
                "rollout_distance": rollout.total_distance,
                "rollout_replans": rollout.total_replans,
                "rollout_avg_risk": rollout.avg_risk,
                "rollout_max_risk": rollout.max_risk,
                "rollout_avg_compute_ms": rollout.avg_compute_ms,
                "rollout_collisions": rollout.collisions,
            }
        )
    return rows


def aggregate(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Return mean and population standard deviation for each planner metric."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["planner"]), []).append(row)

    summary: list[dict[str, object]] = []
    for planner, planner_rows in grouped.items():
        record: dict[str, object] = {"planner": planner, "runs": len(planner_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in planner_rows]
            finite_values = [value for value in values if math.isfinite(value)]
            record[f"{metric}_mean"] = mean(finite_values) if finite_values else float("nan")
            record[f"{metric}_std"] = pstdev(finite_values) if len(finite_values) > 1 else 0.0
        summary.append(record)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, object]]) -> None:
    columns = [
        ("Planner", "planner", "{}"),
        ("Runs", "runs", "{:.0f}"),
        ("Success", "rollout_success_mean", "{:.1%}"),
        ("Distance", "rollout_distance_mean", "{:.2f}"),
        ("Replans", "rollout_replans_mean", "{:.2f}"),
        ("Avg risk", "rollout_avg_risk_mean", "{:.3f}"),
        ("Max risk", "rollout_max_risk_mean", "{:.3f}"),
        ("Compute ms", "rollout_avg_compute_ms_mean", "{:.3f}"),
        ("Collisions", "rollout_collisions_mean", "{:.2f}"),
    ]
    lines = [
        "# DynNav presentation benchmark",
        "",
        "Values are means across identical environment seeds. Runtime values depend on hardware and should not be compared across machines.",
        "",
        "| " + " | ".join(title for title, _, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary:
        cells: list[str] = []
        for _, key, template in columns:
            value = row[key]
            cells.append(str(value) if key == "planner" else template.format(float(value)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "> Synthetic benchmark only. These results are not evidence of certified safety or physical-robot performance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1-20"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/presentation/benchmark"))
    parser.add_argument("--dynamic-step-every", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_SCENARIO.max_steps)
    args = parser.parse_args()

    if args.dynamic_step_every < 1:
        parser.error("--dynamic-step-every must be at least 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    cfg = replace(DEFAULT_SCENARIO, max_steps=args.max_steps)
    rows = [
        row
        for seed in args.seeds
        for row in run_seed(cfg, seed=seed, dynamic_step_every=args.dynamic_step_every)
    ]
    summary = aggregate(rows)

    raw_path = args.out_dir / "raw.csv"
    aggregate_path = args.out_dir / "aggregate.csv"
    markdown_path = args.out_dir / "summary.md"
    write_csv(raw_path, rows, RAW_FIELDS)
    write_csv(aggregate_path, summary, list(summary[0].keys()))
    write_markdown(markdown_path, summary)

    manifest = args.out_dir / "scenario.txt"
    manifest.write_text(
        "\n".join(f"{key}={value}" for key, value in asdict(cfg).items()) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} runs to {args.out_dir}")


if __name__ == "__main__":
    main()
