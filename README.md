# DynNav

**Risk- and recoverability-aware online replanning for autonomous robots in dynamic, partially observed environments.**

[English](README.md) · [Ελληνικά](README_GR.md) · [Project overview](docs/PROJECT_OVERVIEW.md)

[![CI](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml/badge.svg)](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E)](ros2_ws/src/dynnav_nav2_cpp/README.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-4C1.svg)](LICENSE)

## What I built

DynNav is an end-to-end robotics research project for studying one question:

> **Can a robot preserve useful escape and recovery options before a dynamic change makes its current route unusable?**

I built the project as a full research pipeline, not only as a path-planning algorithm. It includes:

- a Python planning and replanning core;
- risk, uncertainty, occupancy, and recoverability models;
- controlled J0–J3 planner ablations;
- paired multi-seed experiments and statistical evaluation;
- a C++17 ROS 2 Jazzy / Nav2 global-planner plugin;
- static and dynamic Gazebo benchmark protocols;
- Streamlit and FastAPI/Next.js research interfaces;
- tests, CI, manifests, failure definitions, and claim–evidence tracking.

**Status:** research prototype. DynNav is not safety certified. The current recoverability quantity is a structural heuristic rather than a calibrated probability of successful recovery.

## Why this matters

A shortest collision-free path is not always a good decision in a changing environment. A robot can enter a narrow corridor, commit to a single-exit region, or lose its last practical retreat option just before a newly observed obstacle invalidates the route.

DynNav therefore evaluates navigation decisions using more than geometric path cost. The controlled planning family uses:

```text
cost(x) = 1 + λr · normalized_risk(x) + λi · irreversibility(x)
```

The goal is to test whether preserving recovery options can reduce **post-invalidation recovery-infeasible failures**, while keeping path-length and computation overhead acceptable.

## Core scientific comparison

All four objectives are evaluated under matched maps, events, configurations, and seeds so that risk and recoverability effects can be separated.

| ID | Objective | What it tests |
|---|---|---|
| J0 | shortest | geometric baseline |
| J1 | shortest + risk | costmap-risk effect |
| J2 | shortest + recoverability penalty | escape-option preservation |
| J3 | shortest + risk + recoverability | joint objective |

The central hypothesis is intentionally **not presented as a proven result**. A sufficiently powered paired dynamic study is still required to establish whether J2 or J3 improves the target failure outcome.

## How the system works

```text
Map / sensor state / dynamic event
               │
               ▼
  Occupancy + risk + uncertainty
               │
               ▼
 Recoverability / escape structure
               │
               ▼
       J0 / J1 / J2 / J3
               │
               ▼
      Python reference or Nav2
               │
        route invalidation
               │
               ▼
             replan
               │
               ▼
 failures + latency + path + risk + artifacts
```

A reportable experiment follows the full loop: **observe → plan → execute → invalidate → replan → evaluate → retain evidence**.

## What is implemented

| Layer | Implementation |
|---|---|
| Planning | Deterministic A*, Dijkstra, J0–J3 recoverability-aware A*, D* Lite experiments |
| Environment | Occupancy, normalized risk, uncertainty, dynamic obstacle updates, safe regions |
| Recoverability | Structural escape-option and irreversibility reasoning |
| Evaluation | Path, risk, failure, overhead, paired-effect and confidence-interval metrics |
| ROS 2 | C++17 `nav2_core::GlobalPlanner` plugin for ROS 2 Jazzy/Nav2 |
| Simulation | Static planner-server and dynamic Gazebo route-invalidation protocols |
| Research tooling | Reproducible runners, manifests, reports, FastAPI/Next.js Researcher, Streamlit lab |
| Extended programme | Exploratory modules across learning, mapping, security, multi-robot and human/AI interaction |

The extended modules are research prototypes and **are not treated as evidence for the central DynNav claim**. The smallest publishable scope is documented in [`CORE_CONTRIBUTION.md`](CORE_CONTRIBUTION.md).

## Evidence snapshot

| Evidence level | Current retained evidence | What it supports |
|---|---|---|
| Software | Python tests, Ruff, strict mapping-core typing | implementation contracts |
| Controlled studies | paired, multi-seed J0–J3 runs and raw artifacts | algorithm debugging and hypothesis refinement |
| Nav2 | retained planner-server runs | path generation / integration evidence |
| Dynamic Gazebo | commissioning trials | experimental pipeline commissioning |
| Physical robot | launch and safety preparation only | no physical-robot efficacy claim |

For exact boundaries, see [`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md).

## Start here if you are reviewing the project

1. **[Project overview](docs/PROJECT_OVERVIEW.md)** — what I built and how the pieces connect.
2. **[Core contribution](CORE_CONTRIBUTION.md)** — the smallest publishable scientific contribution.
3. **[Claim–evidence matrix](CLAIM_EVIDENCE_MATRIX.md)** — what is and is not supported by evidence.
4. **[Experiment protocol V2](EXPERIMENT_PROTOCOL_V2.md)** — experimental design and validity rules.
5. **[Failure cases](FAILURE_CASES.md)** — operational failure definitions and falsification cases.
6. **[Research dossier](docs/PHD_APPLICATION_READINESS.md)** — concise research-review path.

## Repository map

```text
dynnav/                     canonical Python research core
  planners/                 planning and replanning algorithms
  experiments/              controlled scenarios and studies
  evaluation/               metrics and statistical analysis
  mapping/                  mapping / uncertainty components
  researcher/               typed research protocols and reporting

ros2_ws/src/
  dynnav_nav2_cpp/           C++ Nav2 global-planner plugin
  dynnav_nav2_benchmark/     static + dynamic ROS/Gazebo experiments
  dynnav_turtlebot3/         simulation / hardware bring-up

configs/                    experiment configurations
scripts/                    runners, validators, audits
results/                    retained evidence and experiment artifacts
tests/                      regression and research-contract tests
contributions/              exploratory research modules
apps/api/                   FastAPI research API
apps/web/                   Next.js Researcher interface
app/                        Streamlit laboratory
docs/                       scientific and engineering documentation
paper/                      publication-facing material
```

## Reproduce the Python evidence

Requirements: Python 3.10 or newer.

```bash
git clone https://github.com/panagiotagrosdouli/DynNav.git
cd DynNav
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,researcher,dashboard]"
```

Run the fast reproducibility path:

```bash
python scripts/run_all.py \
  --config configs/default.yaml \
  --smoke \
  --out-dir results/ci_smoke

python scripts/run_benchmarks.py \
  --config configs/default.yaml \
  --smoke \
  --out-dir results/ci_benchmarks

python -m pytest -q
```

## ROS 2 Jazzy / Nav2

ROS 2 Jazzy is required only for the Nav2 and Gazebo workflows.

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
```

Experiment definitions are in the [Gazebo benchmark protocol](docs/GAZEBO_BENCHMARK_PROTOCOL.md) and [dynamic execution protocol](docs/DYNAMIC_EXECUTION_PROTOCOL.md).

## Research interfaces

The interfaces configure and inspect the same research pipeline; they do not replace the underlying raw artifacts.

**Researcher API + web workspace**

```bash
python -m uvicorn apps.api.main:app --reload --port 8000
npm --prefix apps/web ci --no-audit --no-fund
npm --prefix apps/web run dev
```

**Streamlit laboratory**

```bash
streamlit run app/dashboard.py
```

## Current limitations and next research step

The current escape-option score has not yet been calibrated against held-out executed recovery outcomes. The retained dynamic study is commissioning evidence rather than a powered treatment comparison, and generalization beyond the retained scenarios has not been established.

The next scientific step is therefore:

**validate a robot-information-conditioned recovery-feasibility estimator → freeze the protocol → run a powered paired dynamic study → analyze failures and overhead → only then extend to staged physical-robot validation.**

## Citation and project policies

- [Citation metadata](CITATION.cff)
- [Contribution guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Publication plan](PUBLICATION_PLAN.md)
- [Reproducibility report](REPRODUCIBILITY_REPORT.md)

DynNav is released under the [Apache License 2.0](LICENSE).
