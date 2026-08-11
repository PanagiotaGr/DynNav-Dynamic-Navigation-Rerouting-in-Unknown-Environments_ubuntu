# Gazebo Harmonic benchmark protocol

## Stage 2 scope

Stage 2 verifies that the DynNav plugin participates in a real ROS 2 Jazzy/Nav2
planner server and produces paths under the same conditions as established
Nav2 planners. It uses the official `nav2_minimal_tb3_sim` Gazebo Harmonic
integration and the `ComputePathToPose` planner selector.

The executable package is
[`dynnav_nav2_benchmark`](../ros2_ws/src/dynnav_nav2_benchmark/README.md).

Implementation status: the launch, runner, schema, package-level tests, and
manual GitHub Actions workflow are committed. No Gazebo measurement is treated
as evidence until that workflow passes on a named commit and its raw artifact is
retained.

## Map and query provenance

The default suite uses the official Navigation2 Jazzy TurtleBot3 sandbox map.
The exact source files inspected when freezing the queries are:

| File | Jazzy git blob |
|---|---|
| `nav2_bringup/maps/tb3_sandbox.yaml` | `676b700a80d7d5479a6eebd62e96a57ac486a7a2` |
| `nav2_bringup/maps/tb3_sandbox.pgm` | `10700a174d9641cd4033ed92b8e5e01b519569e7` |

The map has 0.05 m resolution and origin `[-10, -10, 0]`. Query endpoints were
checked against the PGM (`254` for each endpoint) and the nearest raw occupied
pixel:

| Query endpoint | Pose (m) | Raw occupied-cell clearance (m) |
|---|---:|---:|
| `sandbox_diagonal` start | `(-2.0, -0.5)` | 0.539 |
| `sandbox_diagonal` goal | `(1.75, 1.0)` | 0.453 |
| `sandbox_crossing` start | `(-1.5, 0.5)` | 0.495 |
| `sandbox_crossing` goal | `(1.5, -0.5)` | 0.424 |

This check prevents endpoint collisions in the source map. It does not replace
runtime validation against the inflated Nav2 costmap.

## Experimental controls

- One planner server hosts every compared plugin.
- All planners share one global costmap snapshot stream.
- Start and goal poses are supplied explicitly (`use_start=true`).
- Each repetition is a complete block containing every planner once.
- Planner order uses seeded cyclic counterbalancing; execution-position counts
  differ by at most one for every planner.
- Warm-up calls are excluded from measured trials.
- Raw paths and failures are retained; failures are not dropped from latency or
  success-rate reporting.
- The scenario, complete Nav2 parameter file, map YAML, and map image are copied
  into the artifact directory and identified by SHA-256.
- The installed ROS package versions, Gazebo versions, kernel, source revision,
  and result-generation timestamp are retained with the workflow artifact.

## Stage 2 metrics

- request success rate;
- planning latency in milliseconds;
- path length in meters;
- number of path poses;
- terminal position error in meters.

The artifact contains both aggregate and scenario-stratified summaries. Formal
analysis must use the raw paired trials and retain scenario strata rather than
relying only on pooled means.

These metrics answer engineering integration and static-planning questions.
They do not test the central irreversible-failure hypothesis.

## Stage 3 dynamic-execution implementation

The separate [dynamic route-invalidation protocol](DYNAMIC_EXECUTION_PROTOCOL.md)
implements the next evidence layer and freezes:

1. a time-indexed obstacle-event trace;
2. robot controller and velocity limits;
3. sensor range, update rate, and observation delay;
4. safe recovery regions;
5. a time or distance budget for reaching a safe region;
6. the commitment point after which a route closure is introduced.

Every planner receives the identical event trace. The current primary endpoint
is whether an independent grid path to a predefined safe region still exists on
the post-event inflated costmap within a distance budget. A failed static path
request remains `planning_failure`, not `operational_irreversible_failure`.
This implementation is still awaiting a valid retained Gazebo run and does not
yet include a kinodynamic or collision-contact oracle.

## Evidence tiers

| Tier | Permitted statement |
|---|---|
| Pure C++ tests | The search objective behaves as specified on known grids |
| Nav2 plugin CI | The plugin builds and is discoverable on ROS 2 Jazzy |
| Static Gazebo benchmark | The plugin returns paths through a live Nav2 planner server |
| Dynamic Gazebo benchmark | Comparative route-invalidation outcomes under a frozen protocol |
| TurtleBot3 hardware | Robot-specific outcome under documented safety supervision |

Results must cite the exact source revision and retained artifact directory.

## Platform references

- [Nav2 Gazebo setup for Jazzy/Harmonic](https://docs.nav2.org/setup_guides/gazebo.html)
- [Nav2 planner-server configuration](https://docs.nav2.org/configuration/packages/configuring-planner-server.html)
- [Official minimal TurtleBot3 simulation](https://github.com/ros-navigation/nav2_minimal_turtlebot_simulation)
- [Smac 2D planner configuration](https://docs.nav2.org/configuration/packages/smac/configuring-smac-2d.html)
- [NavFn planner configuration](https://docs.nav2.org/configuration/packages/configuring-navfn.html)
