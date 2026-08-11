# ROS 2 / Nav2 integration

DynNav now contains a real C++ Nav2 global-planner plugin targeting ROS 2
Jazzy on Ubuntu 24.04. The integration boundary is intentionally narrow: the
plugin consumes Nav2's global costmap and returns a collision-checked global
path, while the Python package remains the broader research environment.

## Architecture

```text
Nav2 ComputePathToPose
          |
          v
DynNavGlobalPlanner
          |
          +-- costmap risk proxy
          +-- local escape-option proxy
          |
          v
deterministic four-connected A*
          |
          v
nav_msgs/msg/Path -> Nav2 controller
```

The implementation is in
[`ros2_ws/src/dynnav_nav2_cpp`](../ros2_ws/src/dynnav_nav2_cpp/README.md).
The older `dynnav_nav2` Python node is retained as a diagnostic research bridge;
it is not the Nav2 planner plugin.

## Evidence status

| Capability | Evidence |
|---|---|
| Costmap-backed A* core | Pure C++ compilation and deterministic known-answer tests |
| Jazzy `GlobalPlanner` API | Three-argument `createPlan` with cancellation checker |
| Plugin discovery | pluginlib instantiation test in the ROS CI job |
| ROS 2 Jazzy build | Passing in `ros:jazzy-ros-base-noble` on the evaluated branch |
| Static Gazebo benchmark harness | Implemented with paired requests and provenance artifacts |
| Gazebo Harmonic execution | Passing retained static run with 36/36 successful requests |
| TurtleBot3 simulation | Official minimal simulation exercised in retained static and dynamic runs |
| Dynamic route-invalidation harness | Implemented with Gazebo entity services and a costmap recovery oracle |
| Dynamic result | Passing retained `n=1` commissioning run with 8/8 valid trials; not a powered comparison |
| Physical-robot safety | Not claimed |

The ROS CI result should be cited only after the workflow has completed on the
commit being evaluated.

## Planner contract

For each traversable destination cell `s`, the planner applies

```text
c(s) = c_neutral + lambda_r * R(s) + lambda_irr * I_local(s)
```

where `R(s)` is normalized from the Nav2 costmap value and `I_local(s)` combines
escape-option deficit and bottleneck exposure. The A* heuristic is Manhattan
distance multiplied by `c_neutral`, so the additional non-negative terms do not
invalidate admissibility.

This `I_local` term is a deterministic structural proxy. It must not be reported
as return-failure probability, CVaR, or formal recoverability without a separate
calibration and validation experiment.

## Reproducible build

```bash
source /opt/ros/jazzy/setup.bash
rosdep install \
  --from-paths ros2_ws/src/dynnav_nav2_cpp \
  --ignore-src --rosdistro jazzy -r -y

colcon build \
  --base-paths ros2_ws/src/dynnav_nav2_cpp \
  --packages-select dynnav_nav2_cpp \
  --event-handlers console_direct+
source install/setup.bash

colcon test \
  --packages-select dynnav_nav2_cpp \
  --event-handlers console_direct+
colcon test-result --verbose
```

The standalone grid-search library deliberately has no ROS dependency. This
lets algorithmic tests run independently while ROS CI verifies the ABI,
dependencies, plugin manifest, and runtime discovery.

## Nav2 parameters

```yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 5.0
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

## Next validation milestones

1. Completed: pass Jazzy CI and retain the static and dynamic Gazebo artifacts.
2. Visually inspect the implemented frozen Gazebo obstacle-event timelines.
3. Expand the [dynamic route-invalidation protocol](DYNAMIC_EXECUTION_PROTOCOL.md)
   to paired map/event seeds and compare NavFn, Smac 2D, risk-only DynNav, and
   joint DynNav on identical dynamic runs.
4. Record success, path length, replanning latency, clearances, safety stops, and
   structured failure causes over multiple seeds.
5. Add velocity/dynamics constraints and an independent emergency-stop layer
   before hardware claims.

The Python simulation, ROS integration, Gazebo study, and physical-robot study
must remain separate evidence tiers in papers and PhD application material.
