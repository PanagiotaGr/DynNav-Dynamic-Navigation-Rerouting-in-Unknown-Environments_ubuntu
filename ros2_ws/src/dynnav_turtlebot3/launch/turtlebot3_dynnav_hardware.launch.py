"""Physical TurtleBot3 Nav2 bringup with a merged DynNav planner configuration.

Start the vendor TurtleBot3 base/sensor bringup separately.  This file never
starts Gazebo and always disables simulation time.
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _setup(context):
    base = Path(LaunchConfiguration("base_params_file").perform(context))
    output = Path(LaunchConfiguration("generated_params_file").perform(context))
    params = yaml.safe_load(base.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise RuntimeError(f"invalid base Nav2 parameters: {base}")
    planner = params.setdefault("planner_server", {}).setdefault("ros__parameters", {})
    planner["planner_plugins"] = ["DynNav"]
    planner["DynNav"] = {
        "plugin": "dynnav_nav2_cpp::DynNavGlobalPlanner",
        "allow_unknown": False,
        "lethal_cost_threshold": 253,
        "neutral_cost": 1.0,
        "risk_weight": float(LaunchConfiguration("risk_weight").perform(context)),
        "irreversibility_weight": float(
            LaunchConfiguration("irreversibility_weight").perform(context)
        ),
        "unknown_risk": 0.5,
        "max_iterations": 0,
    }
    for section in params.values():
        if isinstance(section, dict) and isinstance(section.get("ros__parameters"), dict):
            section["ros__parameters"]["use_sim_time"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

    nav2_launch = Path(get_package_share_directory("nav2_bringup")) / "launch" / "bringup_launch.py"
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(nav2_launch)),
            launch_arguments={
                "map": LaunchConfiguration("map"),
                "params_file": str(output),
                "use_sim_time": "False",
                "autostart": LaunchConfiguration("autostart"),
                "use_composition": "False",
            }.items(),
        )
    ]


def generate_launch_description() -> LaunchDescription:
    nav2_share = Path(get_package_share_directory("nav2_bringup"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", description="Absolute path to the known-map YAML"),
            DeclareLaunchArgument(
                "base_params_file", default_value=str(nav2_share / "params" / "nav2_params.yaml")
            ),
            DeclareLaunchArgument(
                "generated_params_file", default_value="/tmp/dynnav_hardware_nav2_params.yaml"
            ),
            DeclareLaunchArgument("risk_weight", default_value="4.0"),
            DeclareLaunchArgument("irreversibility_weight", default_value="4.0"),
            DeclareLaunchArgument("autostart", default_value="true"),
            OpaqueFunction(function=_setup),
        ]
    )
