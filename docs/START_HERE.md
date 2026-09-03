# Start here: understanding DynNav without a robotics background

You do not need to know ROS, Nav2, A*, costmaps or autonomous robotics to understand the purpose of this project.

## The problem in everyday language

Imagine a robot travelling through a building. It sees two possible routes to its destination.

One route is short, but it passes through a narrow area with very few ways to escape. The other route is slightly longer, but gives the robot several alternative directions if a door closes, a person blocks the corridor or the map changes.

A conventional planner can prefer the first route because it is shorter. DynNav asks whether the planner should also care about **what options the robot will still have if its current plan becomes invalid later**.

That is the central idea of the project.

## What was built

DynNav is not one script. It is an end-to-end research system containing:

1. reference planning algorithms in Python;
2. representations of obstacles, risk and local escape options;
3. controlled experiments comparing different planning objectives;
4. metrics and statistical analysis;
5. a C++ planner plugin for ROS 2/Nav2;
6. Gazebo experiments where a route can be invalidated during execution;
7. reproducibility tooling that retains configuration, seeds and raw results;
8. interfaces for inspecting and running experiments;
9. tests and claim/evidence documentation.

## The four planners being compared

The core comparison deliberately changes one idea at a time.

- **J0 — shortest:** mainly asks “which path is shortest?”
- **J1 — risk-aware:** also penalizes risky map regions.
- **J2 — recoverability-aware:** also penalizes states with poor local escape options.
- **J3 — joint:** considers both risk and recoverability.

This structure matters scientifically because it allows the effect of the new recoverability term to be separated from ordinary risk-aware planning.

## What “recoverability” means here

In this repository, recoverability should not be read as a guarantee that the robot can always recover.

The current implementation uses a structural/local estimate of whether a state preserves useful escape or return options. It is a research heuristic. One of the important future steps is to validate or replace it using executed recovery outcomes.

## What happens in an experiment

A typical dynamic experiment follows this sequence:

```text
Robot receives a start and goal
            ↓
Environment/costmap is observed
            ↓
J0, J1, J2 or J3 computes a path
            ↓
Robot begins executing the path
            ↓
An event changes or blocks the route
            ↓
The change becomes visible to the planner
            ↓
The robot replans
            ↓
The experiment records success, failure,
recovery feasibility, path length, risk,
latency and other measurements
```

The project then compares planners across controlled trials rather than judging them from a single attractive path.

## What the repository currently demonstrates

The repository demonstrates implementation, software verification, controlled research experiments, ROS/Nav2 integration and commissioning of a dynamic Gazebo experimental path.

It does **not** currently demonstrate that DynNav is universally safer, that recoverability is a calibrated probability, or that the approach is validated on physical robots. Those distinctions are intentional and documented.

## Where to go next

- Read `README.md` for the complete project explanation and commands.
- Read `docs/PROJECT_OVERVIEW.md` for a reviewer-oriented overview.
- Read `docs/REPOSITORY_MAP.md` to understand the folders.
- Read `CORE_CONTRIBUTION.md` for the smallest publishable scientific claim.
- Read `CLAIM_EVIDENCE_MATRIX.md` to see which claims have which evidence.
- Inspect `dynnav/` for the Python algorithm.
- Inspect `ros2_ws/src/` for the robot integration.
- Inspect `results/` for retained evidence.

If you remember only one sentence, remember this:

> **DynNav studies whether a robot should choose paths not only for where they lead now, but also for the useful options they leave available when the world changes.**
