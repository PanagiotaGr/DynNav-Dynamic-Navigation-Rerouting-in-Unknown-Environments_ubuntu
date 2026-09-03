# DynNav

**Dynamic Navigation: planning paths that preserve options when the world changes.**

[English](README.md) · [Ελληνικά](README_GR.md) · [Start here](docs/START_HERE.md) · [Repository map](docs/REPOSITORY_MAP.md)

[![CI](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml/badge.svg)](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E)](ros2_ws/src/dynnav_nav2_cpp/README.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-4C1.svg)](LICENSE)

## In one sentence

**DynNav studies whether an autonomous robot should choose a route not only because it is short, but also because it leaves useful escape and recovery options available if the environment changes later.**

If terms such as *A\**, *Nav2*, *costmap* or *ROS 2* are unfamiliar, that is fine. This README starts from the problem and introduces the technical pieces gradually. A shorter beginner introduction is also available in [`docs/START_HERE.md`](docs/START_HERE.md).

---

## 1. The problem

Imagine a robot moving through a building from room A to room B.

It has two possible paths:

```text
Path 1: short route through a narrow corridor
Path 2: slightly longer route through a more open area
```

If nothing changes, Path 1 may be the obvious choice. But real environments do change. A person can stop in the corridor, a door can close, an object can appear, or the robot can discover that part of its map was wrong.

Now suppose the robot has already travelled deep into the narrow corridor when the route becomes blocked. The problem is no longer only “find another path to the goal.” The robot may have placed itself in a state from which retreat or recovery is difficult.

This motivates the question behind DynNav:

> **Can the planner consider the future usefulness of the states it enters, instead of optimizing only the route that looks best right now?**

DynNav explores this through **risk-aware** and **recoverability-aware** online replanning.

---

## 2. What I built

DynNav is an end-to-end research project rather than a single path-planning function. The repository contains the complete path from an idea to controlled experiments and robot-stack integration.

I implemented:

1. **A Python research implementation** of planning and replanning algorithms.
2. **Environment models** for occupancy, obstacles, uncertainty and risk.
3. **A recoverability/escape-option signal** that penalizes locally restrictive decisions.
4. **Four controlled planning objectives (J0–J3)** so the effect of risk and recoverability can be separated experimentally.
5. **Dynamic experiments** where the route changes after planning has started.
6. **Metrics and statistical evaluation** for success, failures, path cost, risk, replanning and overhead.
7. **A C++ ROS 2 / Nav2 global planner plugin** so the idea can run inside a standard robotics navigation stack.
8. **Gazebo benchmark infrastructure** for static and dynamic simulation experiments.
9. **Reproducibility tooling** that records configuration, seeds, trial data and reports.
10. **Tests and CI** for software contracts and research tooling.
11. **Research interfaces** using Streamlit and FastAPI/Next.js for configuring and inspecting experiments.
12. **Evidence and research-governance documents** that separate implemented features, observations, hypotheses and unsupported claims.

The repository also contains broader exploratory modules. Those are intentionally separated from the central scientific contribution.

---

## 3. The central idea: preserve options

A normal shortest-path planner asks something close to:

> “How expensive is it to move from here toward the goal?”

DynNav additionally asks:

> “If I move here, am I entering a state with poor escape or recovery options?”

The controlled objective family can be summarized as:

```text
cost = path movement
     + risk penalty
     + irreversibility / poor-recoverability penalty
```

In the implementation this is expressed conceptually as:

```text
cost(x) = 1 + λr · normalized_risk(x) + λi · irreversibility(x)
```

where:

- `1` represents ordinary movement cost;
- `normalized_risk(x)` represents map/cost-related risk at state `x`;
- `irreversibility(x)` penalizes locally restrictive states;
- `λr` controls how strongly risk matters;
- `λi` controls how strongly recoverability matters.

The current recoverability quantity is a **research heuristic**, not a certified probability that recovery will succeed.

---

## 4. Why there are four planners

A research comparison needs to distinguish *which part* of an objective causes a change. For that reason DynNav uses four related objectives.

| Planner | Uses path length | Uses risk | Uses recoverability | Question |
|---|:---:|:---:|:---:|---|
| **J0** | ✓ | — | — | What happens with the geometric baseline? |
| **J1** | ✓ | ✓ | — | What changes when ordinary risk is considered? |
| **J2** | ✓ | — | ✓ | What changes specifically because of recoverability? |
| **J3** | ✓ | ✓ | ✓ | What happens when risk and recoverability are combined? |

This is called an **ablation-style comparison**: components are introduced separately so their effects can be investigated rather than hidden inside one complicated planner.

The project does **not** currently claim that J2 or J3 is universally superior. Establishing that requires a sufficiently powered dynamic experiment under a frozen protocol.

---

## 5. What happens during a DynNav experiment

A simplified experiment looks like this:

```text
1. Create/load an environment
              ↓
2. Choose robot start and goal
              ↓
3. Build the robot-visible map/risk state
              ↓
4. Run J0, J1, J2 or J3
              ↓
5. Robot begins following the route
              ↓
6. Environment changes / route is invalidated
              ↓
7. Robot observes the change
              ↓
8. Planner computes a new route
              ↓
9. Execution succeeds, fails or enters recovery
              ↓
10. Save raw data and compute metrics
```

The important point is that DynNav is interested in **what happens after the original plan stops being valid**, not only in the quality of the first path.

---

## 6. What is measured

Depending on the experiment, the project records quantities such as:

- whether the robot reaches the goal;
- whether the dynamic event was actually observed;
- whether replanning occurred;
- whether a valid recovery remained possible after route invalidation;
- post-invalidation recovery-infeasible failures;
- executed path length;
- planning/replanning latency;
- number of replans;
- cumulative risk score;
- escape-option/recoverability measurements;
- trial validity and failure reasons.

The precise operational definitions for reportable experiments live in [`CORE_CONTRIBUTION.md`](CORE_CONTRIBUTION.md) and [`EXPERIMENT_PROTOCOL_V2.md`](EXPERIMENT_PROTOCOL_V2.md).

---

## 7. From research algorithm to robot software

DynNav has two important implementation levels.

### Python reference layer

The `dynnav/` package is the canonical research implementation. It is useful for controlled experiments because algorithms, scenarios and metrics can be changed and inspected easily.

It contains planning, experiment, mapping, evaluation and research-support components.

### ROS 2 / Nav2 layer

Real robotic navigation software is normally composed of multiple cooperating components. ROS 2 is the middleware ecosystem used here, while Nav2 is the navigation framework.

The repository contains a C++17 `nav2_core::GlobalPlanner` implementation under:

```text
ros2_ws/src/dynnav_nav2_cpp/
```

This allows DynNav-style planning objectives to participate in the same planner-server interface used by Nav2.

The benchmark packages then exercise this integration in simulation, including dynamic route-invalidation experiments.

---

## 8. What Gazebo contributes

Gazebo is used as a robot simulator. Instead of evaluating only abstract grid paths, the project can run a simulated robot and introduce changes while navigation is taking place.

The dynamic workflow is designed to test situations such as:

```text
initial route exists
       ↓
robot starts moving
       ↓
route-changing event occurs
       ↓
change becomes visible in the navigation state
       ↓
original route becomes invalid
       ↓
planner must react
```

This is closer to the research question than testing only a static start-to-goal path.

Simulation evidence is still **simulation evidence**. It is not presented as physical-robot validation.

---

## 9. How the repository fits together

Think of the repository as four layers:

```text
┌────────────────────────────────────────────┐
│  HUMAN / RESEARCH INTERFACES               │
│  app/ · apps/ · docs/                      │
├────────────────────────────────────────────┤
│  EXPERIMENTS AND EVIDENCE                  │
│  configs/ · scripts/ · tests/ · results/   │
├────────────────────────────────────────────┤
│  ROBOT INTEGRATION                         │
│  ros2_ws/src/                              │
├────────────────────────────────────────────┤
│  CORE RESEARCH IMPLEMENTATION              │
│  dynnav/                                   │
└────────────────────────────────────────────┘
```

### Main directories

| Directory | What it contains |
|---|---|
| `dynnav/` | canonical Python planning/research implementation |
| `ros2_ws/src/` | ROS 2, Nav2, Gazebo and TurtleBot3 integration |
| `configs/` | experiment configuration files |
| `scripts/` | reproducible runners, validation and audit tools |
| `tests/` | software and research-contract tests |
| `results/` | retained experiment outputs/evidence |
| `docs/` | explanations, protocols and technical documentation |
| `paper/` | publication-facing material |
| `app/` | Streamlit laboratory |
| `apps/api/` | FastAPI research service |
| `apps/web/` | Next.js research interface |
| `contributions/` | broader exploratory C01–C26 research prototypes |
| `analysis/` | post-processing and investigation scripts |

For a guided explanation of the complete tree, see [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md).

---

## 10. Core work versus exploratory work

A large research repository can easily make prototypes look like one giant scientific claim. DynNav explicitly avoids that.

The **core paper-sized contribution** is the controlled evaluation of risk/recoverability-aware planning under dynamic route invalidation. It is defined in [`CORE_CONTRIBUTION.md`](CORE_CONTRIBUTION.md).

The `contributions/` area contains broader prototypes involving topics such as learning, mapping, multi-robot systems, security and human/AI interaction. These may inform future research, but they are **not evidence that the central DynNav hypothesis has been proven**.

---

## 11. What evidence exists today

The project contains several different evidence levels, and they should not be confused.

| Evidence | What it means |
|---|---|
| Unit/regression tests | software behaves according to tested contracts |
| Controlled Python experiments | algorithms can be compared under controlled scenarios |
| Nav2 planner-server runs | the planner integrates and produces paths through the ROS interface |
| Dynamic Gazebo commissioning | the route-invalidation experimental pipeline can execute in simulation |
| Physical robot | preparation exists, but no physical-robot efficacy result is currently claimed |

For a claim-by-claim view, read [`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md).

---

## 12. What the project does **not** claim

DynNav is a research prototype.

It does not currently claim:

- formal safety guarantees;
- that the recoverability score is a calibrated probability;
- universal superiority over standard planners;
- generalization to arbitrary robots and environments;
- validated physical-robot reliability;
- that every exploratory contribution is part of one proven system.

These boundaries are important because the repository is intended to make the difference between **implementation**, **experimental observation**, **hypothesis** and **future work** visible.

---

## 13. Reproduce the Python side

Requirements: Python 3.10 or newer.

```bash
git clone https://github.com/panagiotagrosdouli/DynNav.git
cd DynNav
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,researcher,dashboard]"
```

Run the fast research pipeline:

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

Run the test suite:

```bash
python -m pytest -q
```

The runners are designed to retain resolved configuration, deterministic seeds, raw trial rows, summaries and reports rather than only printing a final score.

---

## 14. Build the ROS 2 / Nav2 planner

ROS 2 Jazzy is required for this part.

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

See [`docs/GAZEBO_BENCHMARK_PROTOCOL.md`](docs/GAZEBO_BENCHMARK_PROTOCOL.md) and [`docs/DYNAMIC_EXECUTION_PROTOCOL.md`](docs/DYNAMIC_EXECUTION_PROTOCOL.md) for the simulation experiment definitions.

---

## 15. Research interfaces

These interfaces help configure, run and inspect the research pipeline. They are not substitutes for the raw evidence.

### Streamlit laboratory

```bash
streamlit run app/dashboard.py
```

### Researcher API and web workspace

```bash
python -m uvicorn apps.api.main:app --reload --port 8000
npm --prefix apps/web ci --no-audit --no-fund
npm --prefix apps/web run dev
```

---

## 16. Reproducibility philosophy

A useful research result should be traceable back to the conditions that produced it. A reportable DynNav experiment should therefore retain:

- source commit and dirty-tree state;
- exact command;
- resolved configuration;
- deterministic seeds;
- scenario, map, start and goal;
- dynamic event definition;
- raw per-trial outcomes, including failures;
- planner paths and relevant navigation state;
- environment/software versions;
- analysis command and generated report.

The normative efficacy protocol is [`EXPERIMENT_PROTOCOL_V2.md`](EXPERIMENT_PROTOCOL_V2.md).

---

## 17. Current limitations and next scientific step

The current local escape-option score still needs stronger validation against executed recovery outcomes. The retained dynamic simulation work is commissioning evidence rather than a sufficiently powered efficacy comparison. Generalization beyond retained maps, events, seeds and configurations has not been established.

The next scientific path is therefore:

```text
validate recovery-feasibility estimator
                ↓
freeze experiment protocol
                ↓
run powered paired dynamic trials
                ↓
analyze effect + overhead + failures
                ↓
replicate in ROS/Gazebo
                ↓
only then stage physical-robot validation
```

A null or negative result is still scientifically useful if the benchmark, estimator validation and failure analysis are rigorous.

---

## 18. Recommended reading order

### If you know nothing about robotics

1. [`docs/START_HERE.md`](docs/START_HERE.md)
2. this README
3. [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)
4. [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md)

### If you are reviewing the research

1. [`CORE_CONTRIBUTION.md`](CORE_CONTRIBUTION.md)
2. [`EXPERIMENT_PROTOCOL_V2.md`](EXPERIMENT_PROTOCOL_V2.md)
3. [`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md)
4. [`FAILURE_CASES.md`](FAILURE_CASES.md)
5. [`REPRODUCIBILITY_REPORT.md`](REPRODUCIBILITY_REPORT.md)

### If you want the implementation

1. `dynnav/`
2. `tests/`
3. `configs/`
4. `scripts/`
5. `ros2_ws/src/dynnav_nav2_cpp/`
6. `ros2_ws/src/dynnav_nav2_benchmark/`

---

## Citation, contribution and license

- [`CITATION.cff`](CITATION.cff) — citation metadata
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution guide
- [`SECURITY.md`](SECURITY.md) — security policy
- [`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md) — publication path

DynNav is released under the [Apache License 2.0](LICENSE).
