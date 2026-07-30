# DynNav research roadmap

## Research program

DynNav is focused on one primary problem:

> **Planning to preserve escape options during online navigation in dynamic and partially observed environments.**

The project studies whether explicit risk and recoverability terms can prevent a robot from entering states that become difficult or impossible to escape after route invalidation.

## Central research question

> Can explicit recoverability estimation reduce irreversible navigation failures during online replanning without imposing excessive path-length and computation overhead?

## Core hypothesis

A planner that optimizes only current progress or expected occupancy risk may select a route that is locally attractive but structurally fragile. A planner that also estimates future escape and return options should fail less often when the map changes or a route is blocked.

## Canonical objective

```math
J(\pi)=L(\pi)+\lambda_rR(\pi)+\lambda_{irr}I(\pi),
```

where:

- `L` is geometric or traversal cost;
- `R` is cumulative occupancy or collision risk;
- `I` is irreversibility cost;
- `lambda_r` and `lambda_irr` control the trade-off.

A state-level irreversibility model may combine:

```math
I(s)=w_1\frac{1}{1+N_{escape}(s)}+w_2B(s)+w_3P_{return\ failure}(s).
```

The exact definition must be supported by dimensional contracts, normalization, deterministic tests and ablations. No heuristic statistic should be presented as a probability unless it is calibrated and validated as one.

## Experimental hypotheses

- **H1:** Recoverability-aware planning reduces irreversible failure rate compared with shortest-path and risk-only planning.
- **H2:** The benefit increases with the probability and severity of route invalidation.
- **H3:** Failure reduction can be achieved with bounded path-length and runtime overhead.
- **H4:** The combined risk-plus-recoverability objective outperforms either term alone in mixed-risk bottleneck scenarios.

## Required baselines

| Variant | Cost |
|---|---|
| Shortest | `L` |
| Risk-aware | `L + lambda_r R` |
| Recoverability-aware | `L + lambda_irr I` |
| Combined | `L + lambda_r R + lambda_irr I` |

A D* Lite or equivalent incremental replanning baseline should be evaluated with the same scenario sequence, observations and dynamic changes.

## Benchmark scenarios

The benchmark suite will prioritize controlled failure mechanisms:

1. dead ends and single-exit regions;
2. narrow bottlenecks;
3. short fragile route versus long recoverable route;
4. sudden route closure after entry;
5. moving obstacle that blocks return;
6. partial map revelation;
7. repeated block-clear cycles;
8. noisy or delayed occupancy updates.

Each scenario must define the map, start, goal, safe regions, obstacle events, observation schedule and failure condition.

## Primary metrics

The principal metric is:

```math
\text{Irreversible Failure Rate}=
\frac{\text{runs with no feasible safe escape}}
{\text{total runs}}.
```

Supporting metrics:

- mission success rate;
- recovery success rate;
- emergency-stop rate;
- minimum escape-option count;
- bottleneck exposure;
- cumulative risk exposure;
- path-length overhead;
- number of replans;
- planning and replanning time;
- structured failure reason.

## Statistical protocol

Publication-grade experiments should include:

- multiple deterministic seeds;
- identical scenario traces across planners;
- confidence intervals;
- effect sizes;
- paired comparisons where appropriate;
- sensitivity analysis for `lambda_r`, `lambda_irr` and irreversibility subweights;
- failure-case inspection rather than aggregate metrics alone.

## Work packages

### WP1 — Mathematical contracts

- Define units, ranges and normalization for length, risk and irreversibility.
- Define zero-weight behavior.
- Separate scores, probabilities and calibrated probabilities.
- Add configuration validation and deterministic unit tests.

**Exit criterion:** every objective term has an explicit contract and tested disable/ablation behavior.

### WP2 — Recoverability model

- Define safe regions and return targets.
- Implement escape-option counting.
- Implement bottleneck exposure.
- Define returnability and irreversible-failure conditions.
- Add graph and grid test cases with known answers.

**Exit criterion:** the model distinguishes open regions, bottlenecks, dead ends and blocked-return states predictably.

### WP3 — Unified planner interface

- Expose `L`, `R` and `I` through one planner-facing cost API.
- Support the four canonical ablations.
- Preserve deterministic A* and incremental replanning baselines.
- Report structured failure reasons.

**Exit criterion:** all variants run through one configuration and metric schema.

### WP4 — Dynamic route invalidation

- Add deterministic event timelines.
- Support repeated obstacle insertion and clearing.
- Model partial map updates and stale observations.
- Evaluate route invalidation before and after bottleneck entry.

**Exit criterion:** scenario replay produces identical event and planner traces for a fixed seed.

### WP5 — Benchmark runner and artifacts

- Consolidate benchmark commands.
- Store commit, dependency versions, configuration and seeds.
- Generate CSV/JSON results and comparison tables.
- Save event logs and selected failure rollouts.

**Exit criterion:** an external researcher can reproduce a complete comparison from one documented command.

### WP6 — Multi-seed evaluation

- Run the benchmark matrix.
- Produce confidence intervals and effect sizes.
- Generate Pareto plots for safety gain versus path/runtime overhead.
- Document counterexamples and negative results.

**Exit criterion:** H1–H4 can be supported or rejected from stored artifacts.

### WP7 — Integration validation

After the algorithmic and experimental core is stable:

- create a verified ROS 2/Nav2 integration path;
- add deterministic simulation scenarios;
- validate occupancy-grid conversion and path output;
- progress toward physical-robot experiments.

These activities extend the evidence base but do not replace the focused algorithmic evaluation.

## Immediate implementation order

1. Audit existing risk and recoverability implementations.
2. Freeze terminology and mathematical contracts.
3. Add irreversibility unit tests and known-answer maps.
4. Implement the common ablation interface.
5. Build deterministic bottleneck and route-closure scenarios.
6. Add irreversible-failure metrics and structured reasons.
7. Run multi-seed comparisons.
8. Generate paper-ready tables and figures from stored artifacts.

## Secondary repository modules

Learning, multi-robot, semantic, security, language, neural representation and swarm modules remain available as exploratory extensions. They are not milestones for the central research claim and should not delay the work packages above.

See [`CONTRIBUTION_FEATURE_CATALOG.md`](CONTRIBUTION_FEATURE_CATALOG.md).

## Success definition

DynNav succeeds when an external researcher can clone the repository, run the four planner variants on the same dynamic scenarios, reproduce the stored metrics and determine whether preserving escape options measurably reduces irreversible failures.
