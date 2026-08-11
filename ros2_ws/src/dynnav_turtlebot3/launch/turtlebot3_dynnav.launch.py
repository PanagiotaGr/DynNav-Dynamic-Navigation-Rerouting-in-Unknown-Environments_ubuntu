#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Modular DynNav TurtleBot3 bringup.

    Starts the simulation layer first. Nav2 and DynNav planner activation are
    intentionally separated so parameters can be validated independently.
    """

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'turtlebot3_world.launch.py'
            ])
        )
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'bringup_launch.py'
            ])
        )
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'rviz_launch.py'
            ])
        )
    )

    return LaunchDescription([
        # 1. Gazebo start
        gazebo,
        # 2. TurtleBot3 spawn is handled by turtlebot3_gazebo
        # 3. Nav2 params load through bringup configuration
        nav2,
        # 4. DynNav activation comes from Nav2 planner plugin parameters
        # 5. RViz startup
        rviz,
    ])
