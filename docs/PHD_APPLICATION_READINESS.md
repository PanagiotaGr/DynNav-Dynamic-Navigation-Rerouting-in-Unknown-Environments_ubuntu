# DynNav research and engineering dossier

This page is the shortest evidence-first route through DynNav for a research
engineer, robotics lab, graduate admissions committee, or technical reviewer.
It separates demonstrated implementation from scientific claims that still
require experiments.

## Research question and contribution

DynNav studies whether planning that preserves post-invalidation recovery
options can reduce recovery-infeasible failures in dynamic navigation without
unacceptable path-length or computation overhead.

The current defensible contribution is narrower than that full hypothesis: a
Nav2-compatible experimental framework for comparing geometric, costmap-risk,
local escape-option, and joint planning objectives under controlled route
invalidation. Read the exact scope and go/no-go criterion in the
[smallest publishable core](../CORE_CONTRIBUTION.md).

## Ten-minute review path

| Time | Inspect | What it establishes |
|---:|---|---|
| 1 min | [Claim–evidence matrix](../CLAIM_EVIDENCE_MATRIX.md) | Which claims are supported, partial, or unsupported |
| 2 min | [Experiment protocol V2](../EXPERIMENT_PROTOCOL_V2.md) | Fair J0–J3 comparisons, data split, estimands, power plan, and artifact contract |
| 2 min | [Scientific metrics](../dynnav/evaluation/scientific_metrics.py) and [tests](../tests/test_scientific_metrics.py) | Operational failure labels, path/risk integrals, Wilson intervals, and paired effects |
| 2 min | [Nav2 planner plugin](../ros2_ws/src/dynnav_nav2_cpp/) and [dynamic benchmark](../ros2_ws/src/dynnav_nav2_benchmark/) | ROS 2 integration and route-invalidation instrumentation |
| 1 min | [Failure suite](../FAILURE_CASES.md) | Cases designed to falsify or expose harm from the method |
| 2 min | [Retained ROS/Gazebo results](../results/ros2_gazebo/) and [research audit](../RESEARCH_AUDIT.md) | Existing evidence and its limitations |

## Demonstrated capabilities

| Capability | Verifiable evidence | Boundary |
|---|---|---|
| Experimental design | Frozen ablations, validity rules, negative controls, held-out seed ranges, paired estimands | Powered Level-4 evaluation has not been run |
| Scientific software | Typed Python metrics, deterministic planners, tests, CI across Python 3.10–3.12 | Passing tests validate contracts, not real-world efficacy |
| Robotics integration | ROS 2 Jazzy/Nav2 plugin build and discovery; planner-server and Gazebo benchmark harnesses | Physical-robot execution is not evidenced |
| Reproducibility | Machine-readable results, configurations, hashes, environment snapshots, protocolized artifacts | Older exploratory modules do not all meet the V2 artifact contract |
| Research integrity | Explicit unsupported-claim labels, failure taxonomy, falsification cases, submission gates | The local escape-option heuristic still needs construct validation |

## Reproduce the software evidence

From a clean checkout with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,researcher]"
ruff check dynnav ros2_ws/src/dynnav_nav2_benchmark
python -m pytest -q
python scripts/run_all.py --config configs/default.yaml --smoke --out-dir results/ci_smoke
python scripts/run_benchmarks.py --config configs/default.yaml --smoke --out-dir results/ci_benchmarks
```

The separate ROS 2 Jazzy/Gazebo commands and required recorded topics are
defined in the [V2 protocol](../EXPERIMENT_PROTOCOL_V2.md). Do not interpret a
Python-only run as ROS, simulation, or hardware validation.

## Current evidence boundary

- The retained static Gazebo run demonstrates planner-server path requests, not
  closed-loop navigation efficacy.
- The retained dynamic commissioning run has one repetition, eight valid
  trials, and no observed recovery-infeasible failures; it cannot estimate the
  central treatment effect.
- The current local escape-option term is an interpretable structural
  heuristic, not a calibrated probability or formal viability guarantee.
- No named physical-robot run, traceable rosbag, or formal safety certificate
  is present.

These gaps are research tasks, not hidden caveats. The required estimator
study, powered evaluation, ROS rerun, and hardware-readiness gates are listed in
the [publication plan](../PUBLICATION_PLAN.md).

## What this repository currently demonstrates about the author

The repository provides inspectable evidence of robotics software integration,
algorithm implementation, experimental design, statistical measurement,
reproducibility work, failure analysis, and disciplined scientific claims. It
is strongest as a portfolio artifact for research-engineering and robotics
software roles; the next credibility jump will come from completing the frozen
V2 experiment rather than adding unrelated modules.
