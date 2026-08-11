"""Run paired static path queries against multiple plugins in one Nav2 planner server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.parameter import Parameter

from dynnav_nav2_benchmark.analysis import (
    Pose2D,
    TrialRecord,
    balanced_trial_order,
    load_suite,
    path_length,
    resolve_map_image,
    summarize_trials,
    summarize_trials_by_scenario,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_message(navigator: BasicNavigator, pose: Pose2D, frame_id: str) -> PoseStamped:
    message = PoseStamped()
    message.header.frame_id = frame_id
    message.header.stamp = navigator.get_clock().now().to_msg()
    message.pose.position.x = pose.x
    message.pose.position.y = pose.y
    message.pose.orientation.z = math.sin(pose.yaw / 2.0)
    message.pose.orientation.w = math.cos(pose.yaw / 2.0)
    return message


def _run_query(
    navigator: BasicNavigator,
    *,
    scenario_name: str,
    planner_id: str,
    repetition: int,
    order_index: int,
    start: PoseStamped,
    goal: PoseStamped,
) -> TrialRecord:
    started = time.perf_counter_ns()
    error: str | None = None
    try:
        path = navigator.getPath(start, goal, planner_id=planner_id, use_start=True)
    except Exception as exc:  # Nav2 exceptions are transported through the action result.
        path = None
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    if path is None or len(path.poses) < 2:
        return TrialRecord(
            scenario=scenario_name,
            planner_id=planner_id,
            repetition=repetition,
            order_index=order_index,
            success=False,
            planning_latency_ms=latency_ms,
            path_length_m=None,
            pose_count=0 if path is None else len(path.poses),
            goal_error_m=None,
            path_xy=(),
            error=error or "planner returned no valid path",
        )

    points = tuple(
        (float(pose.pose.position.x), float(pose.pose.position.y))
        for pose in path.poses
    )
    last_x, last_y = points[-1]
    goal_error = math.hypot(
        last_x - goal.pose.position.x,
        last_y - goal.pose.position.y,
    )
    return TrialRecord(
        scenario=scenario_name,
        planner_id=planner_id,
        repetition=repetition,
        order_index=order_index,
        success=True,
        planning_latency_ms=latency_ms,
        path_length_m=path_length(points),
        pose_count=len(points),
        goal_error_m=goal_error,
        path_xy=points,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, records: list[TrialRecord]) -> None:
    fields = [
        "scenario",
        "planner_id",
        "repetition",
        "order_index",
        "success",
        "planning_latency_ms",
        "path_length_m",
        "pose_count",
        "goal_error_m",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            row.pop("path_xy")
            writer.writerow(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--params-file", required=True, type=Path)
    parser.add_argument("--map-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--environment", default="ros2_planner_server")
    return parser


def main(argv: list[str] | None = None) -> int:
    parsed, ros_arguments = _parser().parse_known_args(argv)
    if parsed.repetitions <= 0 or parsed.warmup < 0 or parsed.settle_seconds < 0.0:
        raise SystemExit("repetitions must be positive; warmup and settle time non-negative")

    scenario_path = parsed.scenario.resolve()
    params_path = parsed.params_file.resolve()
    map_path = parsed.map_file.resolve()
    map_image_path = resolve_map_image(map_path)
    suite = load_suite(scenario_path)
    output = parsed.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=ros_arguments)
    navigator = BasicNavigator(node_name="dynnav_static_planner_benchmark")
    navigator.set_parameters([Parameter("use_sim_time", value=True)])
    records: list[TrialRecord] = []
    try:
        initial = suite.scenarios[0]
        navigator.setInitialPose(
            _pose_message(navigator, initial.start, initial.frame_id)
        )
        navigator.waitUntilNav2Active()
        time.sleep(parsed.settle_seconds)

        planner_ids = tuple(planner.planner_id for planner in suite.planners)
        warmup_scenario = suite.scenarios[0]
        for planner_id in planner_ids:
            for warmup_index in range(parsed.warmup):
                warmup_record = _run_query(
                    navigator,
                    scenario_name=f"warmup_{warmup_scenario.name}",
                    planner_id=planner_id,
                    repetition=warmup_index,
                    order_index=0,
                    start=_pose_message(
                        navigator, warmup_scenario.start, warmup_scenario.frame_id
                    ),
                    goal=_pose_message(
                        navigator, warmup_scenario.goal, warmup_scenario.frame_id
                    ),
                )
                if not warmup_record.success:
                    navigator.get_logger().warning(
                        f"Warm-up failed for {planner_id}: {warmup_record.error}"
                    )

        for scenario_index, scenario in enumerate(suite.scenarios):
            schedule = balanced_trial_order(
                planner_ids,
                parsed.repetitions,
                suite.seed + scenario_index * 100_000,
            )
            for repetition, block in enumerate(schedule):
                for order_index, planner_id in enumerate(block):
                    record = _run_query(
                        navigator,
                        scenario_name=scenario.name,
                        planner_id=planner_id,
                        repetition=repetition,
                        order_index=order_index,
                        start=_pose_message(navigator, scenario.start, scenario.frame_id),
                        goal=_pose_message(navigator, scenario.goal, scenario.frame_id),
                    )
                    records.append(record)
                    navigator.get_logger().info(
                        f"{scenario.name}/{planner_id}/r{repetition}: "
                        f"success={record.success} latency={record.planning_latency_ms:.3f} ms"
                    )
    finally:
        navigator.destroyNode()
        rclpy.shutdown()

    payload = {
        "schema_version": 1,
        "benchmark_type": "static_compute_path_to_pose",
        "evidence_boundary": (
            "Planner-server path and latency evidence only; no dynamic execution, "
            "irreversible-failure, controller, or physical-robot claim."
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": parsed.environment,
        "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
        "source_revision": os.environ.get("GITHUB_SHA", "unrecorded"),
        "scenario_sha256": _sha256(scenario_path),
        "params_sha256": _sha256(params_path),
        "map_yaml_sha256": _sha256(map_path),
        "map_image_sha256": _sha256(map_image_path),
        "repetitions": parsed.repetitions,
        "warmup_queries_per_planner": parsed.warmup,
        "planners": [asdict(planner) for planner in suite.planners],
        "scenarios": [asdict(scenario) for scenario in suite.scenarios],
        "trials": [record.to_dict() for record in records],
        "summary": summarize_trials(records),
        "summary_by_scenario": summarize_trials_by_scenario(records),
    }
    _atomic_json(output / "results.json", payload)
    _write_csv(output / "trials.csv", records)
    shutil.copyfile(scenario_path, output / "scenario_snapshot.yaml")
    shutil.copyfile(params_path, output / "nav2_params_snapshot.yaml")
    shutil.copyfile(map_path, output / "map_snapshot.yaml")
    shutil.copyfile(map_image_path, output / f"map_image_snapshot{map_image_path.suffix}")

    failed_planners = [
        planner_id
        for planner_id, summary in payload["summary"].items()
        if summary["successes"] == 0
    ]
    if failed_planners:
        print(f"Benchmark completed with planners that never succeeded: {failed_planners}")
        return 2
    print(f"Benchmark complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
