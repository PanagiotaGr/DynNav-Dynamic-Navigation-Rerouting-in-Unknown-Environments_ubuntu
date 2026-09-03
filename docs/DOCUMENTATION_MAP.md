# DynNav documentation map

This is the curated documentation index for the repository. It intentionally points readers to the documents that explain the active project, rather than listing every historical Markdown file equally.

## Start here

- [`../README.md`](../README.md) — complete project explanation from beginner level to reproduction.
- [`START_HERE.md`](START_HERE.md) — zero-background introduction.
- [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) — reviewer-oriented project overview.
- [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md) — what each major repository area is for.

## Core scientific contribution

- [`../CORE_CONTRIBUTION.md`](../CORE_CONTRIBUTION.md) — smallest publishable contribution and operational quantities.
- [`../EXPERIMENT_PROTOCOL_V2.md`](../EXPERIMENT_PROTOCOL_V2.md) — normative efficacy experiment protocol.
- [`../CLAIM_EVIDENCE_MATRIX.md`](../CLAIM_EVIDENCE_MATRIX.md) — claim-to-evidence boundaries.
- [`../FAILURE_CASES.md`](../FAILURE_CASES.md) — failure definitions and falsification cases.
- [`../REPRODUCIBILITY_REPORT.md`](../REPRODUCIBILITY_REPORT.md) — reproducibility status.
- [`../PUBLICATION_PLAN.md`](../PUBLICATION_PLAN.md) — publication path.

## How the system works

- [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) — software/research architecture.
- [`NAVIGATION_PIPELINE.md`](NAVIGATION_PIPELINE.md) — navigation data flow.
- [`MATHEMATICAL_FORMULATION.md`](MATHEMATICAL_FORMULATION.md) — mathematical formulation.
- [`MAPPING.md`](MAPPING.md) — mapping components.
- [`RISK_ESTIMATION.md`](RISK_ESTIMATION.md) — risk modelling.
- [`UNCERTAINTY_MODEL.md`](UNCERTAINTY_MODEL.md) — uncertainty model.

## ROS 2 / Nav2 / Gazebo

- [`ROS2_NAV2_INTEGRATION.md`](ROS2_NAV2_INTEGRATION.md) — integration overview.
- [`../ros2_ws/src/dynnav_nav2_cpp/README.md`](../ros2_ws/src/dynnav_nav2_cpp/README.md) — canonical C++ Nav2 planner plugin.
- [`../ros2_ws/src/dynnav_nav2_benchmark/README.md`](../ros2_ws/src/dynnav_nav2_benchmark/README.md) — benchmark package.
- [`GAZEBO_BENCHMARK_PROTOCOL.md`](GAZEBO_BENCHMARK_PROTOCOL.md) — Gazebo benchmark protocol.
- [`DYNAMIC_EXECUTION_PROTOCOL.md`](DYNAMIC_EXECUTION_PROTOCOL.md) — dynamic route-invalidation protocol.
- [`HARDWARE_VALIDATION_CHECKLIST.md`](HARDWARE_VALIDATION_CHECKLIST.md) — physical-robot validation checklist; not evidence of completed hardware validation.

## Experiments and evidence

- [`../configs/README.md`](../configs/README.md) — experiment configuration guide.
- [`../scripts/README.md`](../scripts/README.md) — runners and validation commands.
- [`../tests/README.md`](../tests/README.md) — testing strategy.
- [`../results/README.md`](../results/README.md) — retained outputs/evidence.
- [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) — benchmark methodology.
- [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) — evaluation methodology.

## Research review and roadmap

- [`PHD_APPLICATION_READINESS.md`](PHD_APPLICATION_READINESS.md) — concise research-review dossier.
- [`RESEARCH_FOCUS.md`](RESEARCH_FOCUS.md) — research focus.
- [`ROADMAP.md`](ROADMAP.md) — development/research roadmap.
- [`../paper/README.md`](../paper/README.md) — manuscript-facing material.

## Exploratory work

The `contributions/` directory contains broader prototypes. Start from [`../contributions/README.md`](../contributions/README.md). These modules are exploratory research directions and should not be interpreted as evidence for the core DynNav claim unless explicitly linked by the claim–evidence matrix.

## Documentation rule

When documents disagree, use this precedence for the active research claim:

```text
CORE_CONTRIBUTION.md
→ EXPERIMENT_PROTOCOL_V2.md
→ CLAIM_EVIDENCE_MATRIX.md
→ retained results/manifests
→ subsystem documentation
→ exploratory/historical notes
```

Historical Markdown files may remain for provenance, but this map is the supported navigation path for new readers.