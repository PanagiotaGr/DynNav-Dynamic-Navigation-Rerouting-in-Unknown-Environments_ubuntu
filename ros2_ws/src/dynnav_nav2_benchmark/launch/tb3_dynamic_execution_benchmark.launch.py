"""Launch Gazebo, Nav2, service bridges, and frozen dynamic execution trials."""

from __future__ import annotations

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from dynnav_nav2_benchmark.configuration import inject_planner_parameters
from dynnav_nav2_benchmark.dynamic_analysis import load_dynamic_suite
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _launch_setup(context):
    base_params = Path(LaunchConfiguration("base_params_file").perform(context))
    generated_params = Path(LaunchConfiguration("generated_params_file").perform(context))
    scenario_path = Path(LaunchConfiguration("scenario_file").perform(context))
    map_file = LaunchConfiguration("map_file").perform(context)
    output = LaunchConfiguration("output_dir").perform(context)
    repetitions = LaunchConfiguration("repetitions").perform(context)
    source_revision = LaunchConfiguration("source_revision").perform(context)
    headless_requested = LaunchConfiguration("headless").perform(context)
    headless = "True" if headless_requested.lower() in {"1", "true", "yes", "on"} else "False"

    suite = load_dynamic_suite(scenario_path)
    payload = yaml.safe_load(base_params.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid Nav2 parameter file: {base_params}")
    try:
        payload = inject_planner_parameters(payload)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    generated_params.parent.mkdir(parents=True, exist_ok=True)
    generated_params.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    nav2_launch = Path(get_package_share_directory("nav2_bringup")) / "launch"
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav2_launch / "tb3_simulation_launch.py")),
        launch_arguments={
            "params_file": str(generated_params),
            "map": map_file,
            "robot_name": suite.robot_entity,
            "headless": headless,
            "use_rviz": "False",
            "use_simulator": "True",
            "use_composition": "False",
            "autostart": "True",
            "use_sim_time": "True",
        }.items(),
    )

    gazebo_services = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="dynnav_gazebo_service_bridge",
        output="screen",
        arguments=[
            f"/world/{suite.world_name}/create@ros_gz_interfaces/srv/SpawnEntity",
            f"/world/{suite.world_name}/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
    )

    rosbag = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "--output", str(Path(output) / "rosbag2"),
            "--storage", "sqlite3",
            "/tf", "/tf_static", "/odom", "/scan", "/cmd_vel", "/amcl_pose",
            "/plan", "/global_costmap/costmap", "/global_costmap/costmap_updates",
            "/local_costmap/costmap", "/local_costmap/costmap_updates",
            "/behavior_tree_log",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_bag")),
    )

    runner = Node(
        package="dynnav_nav2_benchmark",
        executable="dynamic_execution_benchmark",
        name="dynnav_dynamic_execution_benchmark",
        output="screen",
        arguments=[
            "--config", str(scenario_path),
            "--output-dir", output,
            "--repetitions", repetitions,
            "--source-revision", source_revision,
        ],
    )
    delayed_runner = TimerAction(period=10.0, actions=[runner])
    shutdown_after_benchmark = RegisterEventHandler(
        OnProcessExit(
            target_action=runner,
            on_exit=[EmitEvent(event=Shutdown(reason="DynNav dynamic benchmark completed"))],
        )
    )
    return [simulation, gazebo_services, rosbag, delayed_runner, shutdown_after_benchmark]


def generate_launch_description() -> LaunchDescription:
    benchmark_share = Path(get_package_share_directory("dynnav_nav2_benchmark"))
    nav2_share = Path(get_package_share_directory("nav2_bringup"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("base_params_file", default_value=str(nav2_share / "params" / "nav2_params.yaml")),
            DeclareLaunchArgument("generated_params_file", default_value="/tmp/dynnav_dynamic_nav2_params.yaml"),
            DeclareLaunchArgument("scenario_file", default_value=str(benchmark_share / "config" / "sandbox_dynamic_events.yaml")),
            DeclareLaunchArgument("map_file", default_value=str(nav2_share / "maps" / "tb3_sandbox.yaml")),
            DeclareLaunchArgument("output_dir", default_value="/tmp/dynnav_dynamic_benchmark"),
            DeclareLaunchArgument("repetitions", default_value="1"),
            DeclareLaunchArgument("source_revision", default_value="unknown"),
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("record_bag", default_value="true"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
