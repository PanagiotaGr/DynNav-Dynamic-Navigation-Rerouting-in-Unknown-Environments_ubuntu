"""Launch Gazebo Harmonic, Nav2, six planner plugins, and the paired benchmark."""

from __future__ import annotations

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from dynnav_nav2_benchmark.configuration import inject_planner_parameters
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _launch_setup(context):
    base_params = Path(LaunchConfiguration("base_params_file").perform(context))
    generated_params = Path(
        LaunchConfiguration("generated_params_file").perform(context)
    )
    scenario = LaunchConfiguration("scenario_file").perform(context)
    map_file = LaunchConfiguration("map_file").perform(context)
    output = LaunchConfiguration("output_dir").perform(context)
    repetitions = LaunchConfiguration("repetitions").perform(context)
    warmup = LaunchConfiguration("warmup").perform(context)
    headless = LaunchConfiguration("headless").perform(context)

    payload = yaml.safe_load(base_params.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid Nav2 parameter file: {base_params}")
    try:
        payload = inject_planner_parameters(payload)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    generated_params.parent.mkdir(parents=True, exist_ok=True)
    generated_params.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    nav2_launch = Path(get_package_share_directory("nav2_bringup")) / "launch"
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav2_launch / "tb3_simulation_launch.py")),
        launch_arguments={
            "params_file": str(generated_params),
            "map": map_file,
            "headless": headless,
            "use_rviz": "false",
            "use_simulator": "true",
            "use_composition": "false",
            "autostart": "true",
            "use_sim_time": "true",
        }.items(),
    )

    runner = Node(
        package="dynnav_nav2_benchmark",
        executable="static_planner_benchmark",
        name="dynnav_static_planner_benchmark",
        output="screen",
        arguments=[
            "--scenario",
            scenario,
            "--params-file",
            str(generated_params),
            "--map-file",
            map_file,
            "--output",
            output,
            "--repetitions",
            repetitions,
            "--warmup",
            warmup,
            "--environment",
            "gazebo_harmonic_nav2_minimal_tb3",
        ],
    )
    delayed_runner = TimerAction(period=10.0, actions=[runner])
    shutdown_after_benchmark = RegisterEventHandler(
        OnProcessExit(
            target_action=runner,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="DynNav static planner benchmark completed")
                )
            ],
        )
    )
    return [simulation, delayed_runner, shutdown_after_benchmark]


def generate_launch_description() -> LaunchDescription:
    benchmark_share = Path(get_package_share_directory("dynnav_nav2_benchmark"))
    nav2_share = Path(get_package_share_directory("nav2_bringup"))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "base_params_file",
                default_value=str(nav2_share / "params" / "nav2_params.yaml"),
                description="Installed full Nav2 parameters used as the immutable base.",
            ),
            DeclareLaunchArgument(
                "generated_params_file",
                default_value="/tmp/dynnav_nav2_benchmark_params.yaml",
            ),
            DeclareLaunchArgument(
                "scenario_file",
                default_value=str(
                    benchmark_share / "config" / "sandbox_static_queries.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "map_file",
                default_value=str(nav2_share / "maps" / "tb3_sandbox.yaml"),
                description="Exact occupancy map used by simulation and archived.",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="/tmp/dynnav_nav2_benchmark",
            ),
            DeclareLaunchArgument("repetitions", default_value="10"),
            DeclareLaunchArgument("warmup", default_value="1"),
            DeclareLaunchArgument("headless", default_value="true"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
