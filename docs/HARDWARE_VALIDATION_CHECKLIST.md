# Physical TurtleBot3 Validation Checklist

This is a readiness protocol, not evidence that hardware validation occurred.

## Preconditions

- Named robot model and serial/asset identifier recorded.
- ROS 2 Jazzy base, LDS/LiDAR, odometry, TF, and emergency-stop path verified.
- Known static map and AMCL localization validated before enabling navigation.
- `/scan`, `/odom`, `/tf`, `/tf_static`, `/cmd_vel`, `/amcl_pose`, global/local costmaps, and `/plan` have correct frames and timestamps.
- Robot footprint, inflation radius, velocity, acceleration, and controller limits match the physical platform.
- Test area is access-controlled; a human operator holds an independent hardware stop.

## Conservative sequence

1. Wheels raised: verify transforms, scan orientation, stop/cancel, and zero commands.
2. Low-speed teleoperation: verify odometry and footprint clearance.
3. Nav2 J0 (`risk_weight:=0 irreversibility_weight:=0`) on an open route.
4. Repeat J0 to establish controller/localization variance.
5. J1, J2, and J3 one at a time; no dynamic obstacle injection near humans.
6. Only after static acceptance, test a remotely moved lightweight obstacle with a safety observer.

## Launch

Start the vendor base/sensor bringup, then:

```bash
export TURTLEBOT3_MODEL=waffle
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch dynnav_turtlebot3 turtlebot3_dynnav_hardware.launch.py \
  map:=/absolute/path/map.yaml \
  risk_weight:=0.0 \
  irreversibility_weight:=0.0
```

## Evidence capture

Record ROS logs, generated parameter snapshot, map hash, Git SHA/dirty state,
robot/sensor identifiers, calibration, battery state, start/goal, operator,
incident log, and a rosbag containing:

```bash
ros2 bag record -o dynnav_hardware_run \
  /tf /tf_static /odom /scan /cmd_vel /amcl_pose /plan \
  /global_costmap/costmap /global_costmap/costmap_updates \
  /local_costmap/costmap /local_costmap/costmap_updates
```

## Stop criteria

Stop immediately for TF discontinuity, localization jump, stale scan, unexpected
motion, controller oscillation, clearance violation, loss of operator visibility,
or inability to cancel within the tested stop-distance budget. A stopped trial
is retained as a failure/incident, never silently rerun.
