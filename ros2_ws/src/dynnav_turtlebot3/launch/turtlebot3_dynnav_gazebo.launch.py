from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Full DynNav TurtleBot3 Gazebo + Nav2 bringup for ROS 2 Jazzy."""

    turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    nav2_bringup = get_package_share_directory('nav2_bringup')
    dynnav_turtlebot3 = get_package_share_directory('dynnav_turtlebot3')

    use_sim_time = LaunchConfiguration('use_sim_time')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                turtlebot3_gazebo,
                'launch',
                'turtlebot3_world.launch.py'
            )
        )
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav2_bringup,
                'launch',
                'bringup_launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': os.path.join(
                dynnav_turtlebot3,
                'config',
                'nav2_dynnav_params.yaml'
            )
        }.items()
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true'
        ),
        gazebo_launch,
        nav2_launch,
        rviz,
    ])
