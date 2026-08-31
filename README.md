# DynNav

**Risk- and recoverability-aware online replanning for autonomous robots in
dynamic, partially observed environments.**

[English](README.md) · [Ελληνικά](README_GR.md)

[![CI](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml/badge.svg)](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E)](ros2_ws/src/dynnav_nav2_cpp/README.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-4C1.svg)](LICENSE)

DynNav is an experimental robotics platform for studying a specific question:
can a navigation system react to a route invalidation without having committed
the robot to a state from which recovery is no longer feasible?

The repository combines algorithmic prototypes, controlled experiments, a C++
Nav2 planner plugin, Gazebo benchmark harnesses, and evidence-oriented research
interfaces. It is designed to make every important claim traceable to code,
configuration, tests, and retained artifacts.

> **Status:** research prototype. DynNav is not safety certified, and the
> current recoverability quantity is a structural heuristic rather than a
> calibrated probability of successful recovery.

## Research contribution

Conventional global planning typically optimizes geometric length or costmap
cost. DynNav adds a second concern: whether a candidate decision preserves
useful escape and return options after the environment changes.

For a transition to cell `x`, the controlled planning family uses

```text
cost(x) = 1 + λr · normalized_risk(x) + λi · irreversibility(x)
```

The same implementation exposes four ablations so that risk and
recoverability effects can be evaluated independently.

| Objective | Risk weight | Irreversibility weight | Purpose |
|---|---:|---:|---|
| J0 — shortest | 0 | 0 | Geometric baseline |
| J1 — risk-aware | > 0 | 0 | Costmap-risk ablation |
| J2 — recoverability-aware | 0 | > 0 | Structural recovery ablation |
| J3 — joint | > 0 | > 0 | Proposed combined objective |

The central hypothesis is intentionally not stated as a result: a powered,
paired dynamic study is still required to determine whether J2 or J3 reduces
recovery-infeasible failures without unacceptable path-length or computation
overhead.

## What is implemented

| Layer | Current implementation |
|---|---|
| Planning core | Deterministic A*, Dijkstra, J0–J3 recoverability-aware A*, and D* Lite experiments |
| Environment model | Occupancy, normalized risk, uncertainty, dynamic obstacle updates, and safe-region definitions |
| Evaluation | Path, risk, irreversibility, failure, overhead, paired-effect, and confidence-interval metrics |
| ROS integration | C++17 `nav2_core::GlobalPlanner` plugin for ROS 2 Jazzy/Nav2 |
| Simulation | Static planner-server and dynamic Gazebo route-invalidation protocols |
| Research tooling | Reproducible runners, manifests, Markdown reports, FastAPI/Next.js Researcher, and Streamlit lab |
| Extended work | Registered C01–C26 prototypes covering learning, mapping, multi-robot systems, security, and human/AI interaction |

The canonical execution path is:

1. observe the occupancy and costmap state;
2. estimate risk and local recovery structure;
3. compute a J0–J3 path;
4. execute through Nav2 or the Python reference environment;
5. apply or perceive a route-changing event;
6. replan and retain the full evidence bundle.

## Evidence snapshot

| Evidence level | Retained result | Interpretation |
|---|---|---|
| Software contracts | Python 3.10–3.12 tests, Ruff, strict mapping-core typing | Validates implementation contracts, not navigation efficacy |
| Controlled Python studies | Paired, multi-seed J0–J3 protocol and raw artifacts | Suitable for algorithm debugging and hypothesis refinement |
| Nav2 planner-server | 36/36 successful retained static requests across six planners | Demonstrates path generation on two retained queries |
| Dynamic Gazebo commissioning | 8 valid trials, 7 successful executions | Confirms the experimental path; sample size is insufficient for treatment claims |
| Physical robot | Launch and safety checklist only | No traceable hardware execution is currently claimed |

For the exact status of every claim, use the
[claim–evidence matrix](CLAIM_EVIDENCE_MATRIX.md). The
[research dossier](docs/PHD_APPLICATION_READINESS.md) provides a concise review
path for laboratories, admissions committees, and research-engineering teams.

## Reproduce the software evidence

Requirements: Python 3.10 or newer. ROS 2 Jazzy is required only for the Nav2
and Gazebo workflows.

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
```

Verify the checkout:

```bash
ruff check dynnav ros2_ws/src/dynnav_nav2_benchmark
mypy dynnav/mapping --strict --no-warn-unused-ignores --show-error-codes
python -m pytest -q
python scripts/audit_markdown.py --root . --json-out results/markdown_audit.json
```

Each runner records the resolved configuration, seeds, raw rows, summaries, and
reports in the selected output directory. Failed and partial trials remain
visible.

## ROS 2 Jazzy / Nav2

Build and test the canonical plugin:

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
colcon test-result --verbose
```

The plugin snapshots the global costmap under its mutex, validates frames and
bounds, supports cancellation, rejects lethal goals, and returns a stamped
`nav_msgs/msg/Path`.

Experiment definitions and interpretation rules are documented in the
[Gazebo benchmark protocol](docs/GAZEBO_BENCHMARK_PROTOCOL.md) and the
[dynamic execution protocol](docs/DYNAMIC_EXECUTION_PROTOCOL.md).

## Research interfaces

The interfaces are optional views over the same research contracts; they do
not replace the raw artifacts.

### DynNav Researcher

```bash
python -m uvicorn apps.api.main:app --reload --port 8000
npm --prefix apps/web ci --no-audit --no-fund
npm --prefix apps/web run dev
```

Open `http://localhost:3000`. A research request is converted into an editable,
typed protocol and requires explicit confirmation before execution.

### Streamlit laboratory

```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` for scenario construction, planner comparison,
mapping inspection, experiment control, and result replay.

## Repository guide

```text
dynnav/                     canonical Python package
  planners/                 J0–J3 search and incremental replanning
  experiments/              scenarios and controlled studies
  evaluation/               metrics and statistical summaries
  researcher/               typed protocols and reporting

ros2_ws/src/
  dynnav_nav2_cpp/          C++17 Nav2 global-planner plugin
  dynnav_nav2_benchmark/    static and dynamic Gazebo experiments
  dynnav_turtlebot3/        simulation and hardware bringup

apps/api/                   FastAPI research API
apps/web/                   Next.js Researcher workspace
app/                        Streamlit laboratory
contributions/              C01–C26 exploratory modules
configs/                    experiment configurations
scripts/                    runners, validators, and audits
tests/                      regression and research-contract tests
results/                    retained evidence and generated artifacts
docs/                       scientific and engineering documentation
paper/                      manuscript planning material
```

## Reproducibility contract

A reportable experiment should retain:

- source commit and dirty-tree state;
- exact command, configuration, and deterministic seeds;
- map, scenario, start/goal, and event definitions;
- raw per-trial data, including failures and invalid trials;
- planner paths, costmaps, timestamps, and ROS logs where applicable;
- environment and package versions;
- artifact hashes and the analysis command.

The [V2 protocol](EXPERIMENT_PROTOCOL_V2.md) is normative for new efficacy
experiments.

## Current limitations

- The local escape-option score has not been calibrated against held-out,
  executed recovery outcomes.
- The retained dynamic study is commissioning evidence, not a powered efficacy
  comparison.
- Generalization beyond the retained maps, events, seeds, and configurations
  has not been established.
- No physical-robot reliability result or formal verification artifact is
  currently available.

These limitations define the next research work: estimator validation, a
powered preregistered dynamic study, complete ROS reruns with immutable
manifests, and only then a staged TurtleBot3 experiment.

## Contributing and citation

- [Contribution guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Citation metadata](CITATION.cff)
- [Publication plan](PUBLICATION_PLAN.md)

DynNav is released under the [Apache License 2.0](LICENSE).
