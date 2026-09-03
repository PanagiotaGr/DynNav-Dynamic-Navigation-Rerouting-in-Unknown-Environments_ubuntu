# DynNav — Project Overview

## What I built

DynNav is a research-grade autonomous navigation project focused on **dynamic replanning under uncertainty**.

The central idea is simple: a robot should not only choose a short path; it should also avoid decisions that leave it with no useful recovery option if the environment changes.

I built the project as a complete research pipeline rather than a single planner implementation:

1. **Planning algorithms** — deterministic A*, Dijkstra, D* Lite, and a controlled J0–J3 family that separates geometric cost, risk, and recoverability.
2. **Dynamic environment models** — occupancy, uncertainty, risk, obstacle updates, and safe-region definitions.
3. **Recoverability reasoning** — structural estimates of whether a navigation decision preserves escape and return options.
4. **Evaluation tooling** — paired multi-seed experiments, failure metrics, latency and path-cost measurements, confidence intervals, and retained raw artifacts.
5. **ROS 2 / Nav2 integration** — a C++17 `nav2_core::GlobalPlanner` plugin and benchmark packages.
6. **Gazebo experiments** — static planner-server checks and dynamic route-invalidation protocols.
7. **Research interfaces** — a Streamlit laboratory plus FastAPI/Next.js tools for configuring, running, and inspecting experiments.
8. **Reproducibility and evidence** — CI, tests, frozen configurations, manifests, claim–evidence tracking, and publication-oriented protocols.

## The research question

The project asks:

> When a route becomes invalid during execution, can a planner reduce recovery-infeasible failures by preserving useful escape options before the failure occurs?

This differs from classical shortest-path planning because the cheapest path can still be brittle. A robot may enter a narrow region or consume its last safe retreat option immediately before a new obstacle appears.

## The controlled comparison

The core scientific comparison uses four objectives implemented under the same experimental conditions:

| ID | Objective | Meaning |
|---|---|---|
| J0 | shortest path | geometric baseline |
| J1 | path + risk | risk-aware baseline |
| J2 | path + recoverability penalty | tests escape-option preservation |
| J3 | path + risk + recoverability penalty | joint objective |

The goal is not to claim that J2 or J3 is universally better. The goal is to measure **when preserving recovery options helps, what it costs, and when it fails**.

## How the system works

```text
Map / sensor state / obstacle event
                │
                ▼
 Occupancy + risk + uncertainty
                │
                ▼
 Recoverability / escape-option estimate
                │
                ▼
      J0 / J1 / J2 / J3 planner
                │
                ▼
      Execute through Python or Nav2
                │
        environment changes
                │
                ▼
             Replan
                │
                ▼
 Metrics + failures + raw evidence bundle
```

A normal experiment therefore follows the full loop: observe → plan → execute → invalidate route → replan → measure outcome.

## What is core and what is exploratory

The **core DynNav work** is the risk/recoverability-aware replanning pipeline, its controlled J0–J3 comparison, ROS 2/Nav2 integration, and evidence-oriented evaluation.

The repository also contains exploratory modules covering learning, mapping, security, multi-robot coordination, human/AI interaction, and other research directions. These are useful prototypes, but they are **not evidence for the central DynNav claim**.

This distinction is intentional: the repository separates the smallest publishable scientific contribution from the broader research programme.

## Evidence currently available

The repository contains several levels of evidence:

- Python regression and research-contract tests.
- Reproducible controlled synthetic experiments.
- ROS 2 / Nav2 plugin and integration tests.
- Retained static planner-server runs.
- Dynamic Gazebo commissioning trials.
- Experiment protocols, manifests, failure definitions, and claim–evidence documentation.

The current dynamic Gazebo evidence is commissioning-level rather than a powered efficacy study. The project therefore does **not** claim formal safety guarantees, universal performance superiority, or physical-robot validation.

## Repository reading path

For a first review, use this order:

1. [`README.md`](../README.md) — project summary and quick start.
2. [`CORE_CONTRIBUTION.md`](../CORE_CONTRIBUTION.md) — the smallest publishable scientific contribution.
3. [`CLAIM_EVIDENCE_MATRIX.md`](../CLAIM_EVIDENCE_MATRIX.md) — what each claim is supported by.
4. [`EXPERIMENT_PROTOCOL_V2.md`](../EXPERIMENT_PROTOCOL_V2.md) — experimental design and validity rules.
5. [`FAILURE_CASES.md`](../FAILURE_CASES.md) — failure definitions and falsification cases.
6. [`REPRODUCIBILITY_REPORT.md`](../REPRODUCIBILITY_REPORT.md) — reproducibility status.
7. [`docs/PHD_APPLICATION_READINESS.md`](PHD_APPLICATION_READINESS.md) — concise research-review dossier.

## Main code paths

```text
dynnav/                     Python research core
  planners/                 planning and replanning algorithms
  experiments/              controlled scenarios and studies
  evaluation/               metrics and statistical analysis

ros2_ws/src/
  dynnav_nav2_cpp/           C++ Nav2 global-planner plugin
  dynnav_nav2_benchmark/     static and dynamic ROS/Gazebo benchmarks
  dynnav_turtlebot3/         simulation / hardware bring-up

configs/                    frozen experiment configuration
scripts/                    reproducibility runners and validators
tests/                      software and research-contract tests
results/                    retained evidence and experiment artifacts
contributions/              exploratory research modules
```

## Current research boundary

The present recoverability quantity is a structural heuristic, not a calibrated probability that recovery will succeed. A major next step is validating a robot-information-conditioned recovery-feasibility estimator and then running a sufficiently powered paired dynamic study.

That boundary is important: DynNav is designed to make unsupported claims difficult by keeping code, experiments, limitations, and evidence explicitly linked.
