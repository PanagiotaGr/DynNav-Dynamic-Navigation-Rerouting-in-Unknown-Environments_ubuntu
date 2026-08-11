from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Initial DynNav TurtleBot3 ROS2 Jazzy bringup.

    This launch file is the integration point for Gazebo, TurtleBot3,
    Nav2 and the DynNav planner plugin.
    """

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])
