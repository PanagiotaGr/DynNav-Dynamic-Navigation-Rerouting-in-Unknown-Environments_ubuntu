# DynNav Nav2 benchmark package

This ROS 2 Jazzy package compares six global-planner configurations through
Nav2's `ComputePathToPose` action while holding the map, costmap, start/goal
queries, and trial schedule fixed.

## Compared planners

| ID | Implementation | Objective role |
|---|---|---|
| `NavFn` | NavFn Dijkstra | Established geometric baseline |
| `Smac2D` | Smac 2D A* | Established cost-aware baseline |
| `DynNavShortest` | DynNav | Zero-weight ablation |
| `DynNavRisk` | DynNav | Costmap-risk-only ablation |
| `DynNavRecoverability` | DynNav | Local escape-option-only ablation |
| `DynNavJoint` | DynNav | Joint risk and local irreversibility |

Planner order uses seeded cyclic counterbalancing within complete blocks, so
each planner's execution-position counts differ by at most one. Every plugin
receives the same explicit start and goal on the same planner-server costmap.

## Build

```bash
source /opt/ros/jazzy/setup.bash
rosdep install \
  --from-paths \
    ros2_ws/src/dynnav_nav2_cpp \
    ros2_ws/src/dynnav_nav2_benchmark \
  --ignore-src --rosdistro jazzy -r -y

colcon build \
  --base-paths \
    ros2_ws/src/dynnav_nav2_cpp \
    ros2_ws/src/dynnav_nav2_benchmark \
  --packages-select dynnav_nav2_cpp dynnav_nav2_benchmark
source install/setup.bash
```

## Run with Gazebo Harmonic

```bash
ros2 launch dynnav_nav2_benchmark \
  tb3_static_planner_benchmark.launch.py \
  headless:=true \
  repetitions:=10 \
  output_dir:=$PWD/results/nav2_static
```

The launch file derives a complete parameter file from the installed Jazzy
`nav2_bringup/params/nav2_params.yaml`, replaces only the planner-server plugin
map, launches the official minimal TurtleBot3 simulation, and preserves the
generated parameter snapshot with the results.

The default endpoints are frozen against the official Jazzy sandbox map and
their raw occupied-cell clearances are recorded in the [benchmark
protocol](../../../docs/GAZEBO_BENCHMARK_PROTOCOL.md).

Outputs:

- `results.json`: environment, hashes, raw paths, raw trials, and summaries;
- `trials.csv`: one row per planner request;
- `scenario_snapshot.yaml`: exact paired query definition;
- `nav2_params_snapshot.yaml`: complete parameters used by Nav2;
- `map_snapshot.yaml` and `map_image_snapshot.pgm`: exact occupancy map and
  hashes used by the launch.

The manual `ROS 2 Gazebo benchmark` GitHub workflow also retains installed ROS
and Gazebo versions. A configured workflow is not a completed experiment: cite
only an artifact produced by a passing run on the exact source commit.

## Run frozen dynamic execution trials

```bash
ros2 launch dynnav_nav2_benchmark \
  tb3_dynamic_execution_benchmark.launch.py \
  headless:=true \
  repetitions:=1 \
  output_dir:=$PWD/results/nav2_dynamic
```

The dynamic runner executes Nav2 goals for NavFn, Smac 2D, and all four DynNav
J0--J3 ablations, moves a physical blocker through Gazebo's entity services,
and saves the post-event global costmap for an independent safe-region
reachability assessment. It also checkpoints partial JSON/CSV after every
trial. See the [dynamic protocol](../../../docs/DYNAMIC_EXECUTION_PROTOCOL.md)
for the exact failure taxonomy and evidence limits.

## Scientific boundary

This is a static global-path and latency benchmark. Gazebo supplies a live ROS 2
system and costmap, but the robot is not commanded to execute each path and no
dynamic obstacle is injected. Consequently, these results cannot measure
collision rate, recovery success, controller performance, or irreversible
failure. Those require the subsequent dynamic-execution protocol.

The separate dynamic runner implements that next protocol, but implementation
alone is not a result. Dynamic claims require a passing retained workflow
artifact with every trial marked valid.
