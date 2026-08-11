# DynNav Nav2 Global Planner

`dynnav_nav2_cpp` is a ROS 2 Jazzy/Nav2 global-planner plugin backed by a
deterministic, costmap-aware A* search.

## Implemented behavior

- implements the Jazzy `nav2_core::GlobalPlanner` lifecycle and cancellation API;
- snapshots the Nav2 global costmap under its mutex;
- validates frames and start/goal map bounds;
- rejects lethal goals and never expands lethal cells;
- handles unknown cells according to `allow_unknown`;
- returns a stamped `nav_msgs/msg/Path` in the global costmap frame;
- exposes risk and local-irreversibility weights as plugin parameters;
- raises the standard Nav2 planner exceptions for invalid requests;
- includes deterministic grid-search tests and a pluginlib discovery test.

For a transition into cell `s`, the current implementation minimizes

```text
c(s) = neutral_cost
     + risk_weight * normalized_costmap_cost(s)
     + irreversibility_weight * local_irreversibility(s)
```

`local_irreversibility` combines the deficit in traversable four-connected
escape options with a bottleneck penalty. It is a structural heuristic in
`[0, 1]`; it is not a calibrated probability or a formal viability guarantee.

## Supported platform

- Ubuntu 24.04
- ROS 2 Jazzy
- Nav2 Jazzy
- C++17

The repository CI builds this package in `ros:jazzy-ros-base-noble`, runs the
known-answer search tests, and verifies pluginlib discovery. Gazebo and robot
validation are separate evidence milestones.

## Build and test

```bash
source /opt/ros/jazzy/setup.bash
rosdep install \
  --from-paths ros2_ws/src/dynnav_nav2_cpp \
  --ignore-src --rosdistro jazzy -r -y

colcon build \
  --base-paths ros2_ws/src/dynnav_nav2_cpp \
  --packages-select dynnav_nav2_cpp
source install/setup.bash

colcon test --packages-select dynnav_nav2_cpp
colcon test-result --verbose
```

## Nav2 configuration

Merge [`config/nav2_params.yaml`](config/nav2_params.yaml) into the robot's Nav2
parameters. The required planner-server fragment is:

```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["DynNav"]
    DynNav:
      plugin: "dynnav_nav2_cpp::DynNavGlobalPlanner"
      allow_unknown: true
      lethal_cost_threshold: 253
      neutral_cost: 1.0
      risk_weight: 4.0
      irreversibility_weight: 4.0
      unknown_risk: 0.5
      max_iterations: 0
```

`max_iterations: 0` means unlimited. The Manhattan heuristic remains
admissible because every transition costs at least `neutral_cost`.

## Evidence boundary

This plugin uses one costmap snapshot per planning request. Replanning is
triggered by Nav2, not by an internal background loop. The present
irreversibility term is local, and the plugin does not yet ingest learned
uncertainty, predicted obstacle trajectories, or a robot-specific dynamics
model. Those capabilities require their own interfaces, ablations, and tests.
