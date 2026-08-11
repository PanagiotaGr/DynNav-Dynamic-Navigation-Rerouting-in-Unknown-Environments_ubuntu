from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """DynNav TurtleBot3 Gazebo bringup scaffold for ROS 2 Jazzy."""

    turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                turtlebot3_gazebo,
                'launch',
                'turtlebot3_world.launch.py'
            )
        )
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        rviz,
    ])
