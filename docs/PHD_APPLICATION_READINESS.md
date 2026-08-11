# PhD application readiness

DynNav should be presented as a focused research artifact about preserving
escape options during dynamic navigation, not as a collection of unrelated
robotics modules. A competitive application needs a traceable chain from a
research question to code, controlled experiments, uncertainty-aware analysis,
and honest evidence boundaries.

## Current engineering baseline

| Item | Current status | Evidence |
|---|---|---|
| Canonical installable Python package | Implemented | Wheel contains mapping, experiments, and researcher modules |
| Python regression suite | Implemented | 501 passing and 32 environment-dependent skips locally on Python 3.12; commit CI pending |
| Static benchmark failure semantics | Corrected | Planning failure is no longer labeled irreversible failure |
| Synthetic pipeline artifacts | Corrected | Planned paths are stored; plots represent their stated signals |
| Costmap-backed Nav2 planner | Implemented | C++17 A* core plus Jazzy `GlobalPlanner` adapter |
| Standalone C++ validation | Implemented | Strict compilation and known-answer executable tests |
| ROS 2 Jazzy build and plugin discovery | CI configured | Must pass on the exact pushed commit |
| Static Gazebo benchmark harness | Implemented | Six planners, paired blocks, raw records, hashes, and environment manifest |
| Gazebo Harmonic execution | CI configured | Manual workflow must pass on the exact commit; no result claimed yet |
| TurtleBot3 simulation | CI configured | Official minimal TB3 launch; no retained run claimed yet |
| Dynamic route-invalidation harness | Implemented | Frozen events, real Nav2 execution, costmap oracle, raw traces, invalid-trial guards |
| Dynamic experimental result | Pending | No quantitative claim until a passing retained artifact exists |
| Physical-robot experiment | Pending | No claim yet |

Local test counts are development evidence, not a permanent scientific result.
For applications or papers, link the passing workflow for the exact commit.

## Primary claim

Use one central claim:

> Explicitly penalizing loss of local escape options can reduce irreversible
> failures after route invalidation, with measurable path-length and runtime
> trade-offs.

The existing local Nav2 term is a structural heuristic. H1 is not established
until dynamic execution experiments measure whether safe retreat is actually
lost. A static `no path` result is only a planning failure.

## Required experiment matrix

Run every method on identical maps, starts, goals, sensor observations, and
obstacle-event traces.

| Dimension | Minimum defensible design |
|---|---|
| Planners | NavFn, Smac 2D, DynNav risk-only, DynNav joint risk + irreversibility |
| Scenarios | open space, dual corridor, bottleneck, post-commitment closure, moving return blockage |
| Seeds | At least 30 paired seeds per scenario; increase after power analysis |
| Platforms | Python controlled benchmark, Gazebo Harmonic, TurtleBot3 Burger simulation |
| Primary endpoint | Irreversible failure under an explicit safe-exit definition |
| Secondary endpoints | success, path length, clearance, cumulative costmap exposure, replans, latency, safety stops |
| Statistics | paired effects, confidence intervals, sensitivity analysis, and failure-case inspection |

Do not pool fundamentally different failure mechanisms without also reporting
scenario-stratified results.

## Definition required before data collection

For a dynamic execution state `s_t`, define a safe recovery set `S_safe`, the
current traversability model, a time/distance budget, and robot dynamics. Then
record an irreversible failure only when no admissible trajectory from `s_t`
reaches `S_safe` within that contract. Store the first failure timestep and its
cause. This definition must be frozen before comparing planners.

## Remaining application-grade exit criteria

1. The Jazzy plugin CI and manual Gazebo workflow pass on the application commit.
2. The generated Gazebo artifact, parameter snapshots, and installed-package
   manifest are archived with a stable identifier.
3. Dynamic scenario event traces are serialized and replayed identically across planners.
4. The dynamic four-way ablation runs from one command and writes raw per-seed records.
5. Every reported figure is generated from those raw records by a committed script.
6. The report includes confidence intervals, effect sizes, negative results, and
   at least two inspected failure cases.
7. A two-to-three minute video shows the same scenario under a baseline and
   DynNav, with the map, costmap, global path, and failure reason visible.

The first committed dynamic suite is a smoke protocol, not the final powered
study. Its two frozen events must be visually inspected after a valid run, then
expanded to paired map/event seeds before use as application evidence.

## Application package

The strongest portfolio bundle is small and coherent:

- one-page research summary with question, gap, method, result, and limitation;
- repository link pinned to a passing commit or release;
- reproducibility command and archived raw results;
- short comparison video;
- technical report or preprint with the four-way ablation;
- a concise statement of the next theoretical step, such as a non-local
  returnability estimator or viability approximation.

The additional learning, security, multi-robot, language, and mapping modules
can be mentioned as breadth, but they should not compete with the primary
navigation contribution for attention.
