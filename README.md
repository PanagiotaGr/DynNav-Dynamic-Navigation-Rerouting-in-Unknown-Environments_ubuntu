<div align="center">

# DynNav

## Planning to Preserve Escape Options

**Risk- and recoverability-aware online replanning in partially observed, dynamically changing environments.**

[![CI](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml/badge.svg)](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache--2.0-4C1.svg)](LICENSE)

[Documentation](docs/README.md) · [Extended research modules](docs/CONTRIBUTION_FEATURE_CATALOG.md) · [Dashboard](app/README.md) · [Research roadmap](docs/DYNNAV_V2_RESEARCH_ROADMAP.md)

</div>

---

## Research focus

Autonomous navigation is not only the problem of finding a collision-free or short path to a goal. In an unknown or changing environment, a locally attractive decision may place the robot in a state from which safe retreat, recovery, or replanning is no longer feasible.

DynNav studies one focused question:

> **Can explicit recoverability estimation reduce irreversible navigation failures during online replanning without imposing excessive path-length and computation overhead?**

The project develops and evaluates navigation methods that jointly reason about:

- geometric path cost;
- occupancy and traversal risk;
- preservation of escape and return options;
- route invalidation caused by newly observed or dynamic obstacles;
- online replanning after the environment changes.

The central principle is:

> **A planner should evaluate not only whether a decision moves the robot toward the goal, but also whether that decision preserves enough safe options for reacting when the environment changes.**

---

## Core contribution

DynNav combines three capabilities into one experimental navigation framework:

1. **Risk-aware planning** — avoid routes with excessive collision or occupancy exposure.
2. **Recoverability-aware planning** — penalize states that reduce safe retreat, escape, or future replanning options.
3. **Dynamic replanning** — update the route when observations or obstacles invalidate the current plan.

The intended planner objective is:

```text
J(path) = length(path)
        + lambda_risk * risk(path)
        + lambda_irr  * irreversibility(path)
```

where:

- `length(path)` measures geometric or traversal cost;
- `risk(path)` measures cumulative or local occupancy-related exposure;
- `irreversibility(path)` measures the loss of safe future options.

A state-level irreversibility model may use measurable quantities such as:

```text
I(state) = w_escape     / (1 + escape_options(state))
         + w_bottleneck * bottleneck_exposure(state)
         + w_return     * return_failure_probability(state)
```

The exact formulation, normalization, and weights are treated as experimental variables rather than universal constants.

---

## Scientific hypotheses

The project evaluates the following hypotheses:

- **H1:** Recoverability-aware planning reduces irreversible navigation failures relative to shortest-path and risk-only planning.
- **H2:** Its benefit increases as route invalidation and environmental change become more likely.
- **H3:** The reduction in failures can be achieved with bounded path-length and planning-time overhead.
- **H4:** Joint risk and recoverability reasoning outperforms either component used in isolation.

These hypotheses are evaluated through controlled scenarios, repeated seeded experiments, ablations, sensitivity analysis, and explicit failure reporting.

---

## Experimental comparison

DynNav compares four planner configurations under identical maps, seeds, starts, goals, and obstacle events.

| Configuration | Objective |
|---|---|
| **Shortest** | geometric path cost only |
| **Risk-aware** | length + occupancy risk |
| **Recoverability-aware** | length + irreversibility |
| **Joint planner** | length + risk + irreversibility |

The minimum ablation set is:

```text
J0 = L
J1 = L + lambda_risk * R
J2 = L + lambda_irr  * I
J3 = L + lambda_risk * R + lambda_irr * I
```

Each objective term must have documented units or normalization, value ranges, zero-weight behavior, logging semantics, and deterministic tests.

---

## Evaluation scenarios

The core benchmark suite focuses on environments where shortest-path reasoning can produce brittle or irreversible decisions:

- dead ends and cul-de-sacs;
- narrow passages and bottlenecks;
- two-corridor maps with unequal recovery options;
- sudden route closure after the robot commits to a corridor;
- moving obstacles that block retreat;
- partially observed maps revealed during execution;
- repeated block-and-clear cycles during online replanning.

A representative stress test is:

> The shortest route enters a narrow corridor. After commitment, a new observation or dynamic obstacle blocks the exit. A longer alternative route would have preserved multiple escape and recovery options.

---

## Primary metrics

The main evaluation metrics are:

- mission success rate;
- irreversible failure rate;
- recovery success rate;
- emergency-stop rate;
- minimum escape-option count;
- number of replans;
- path-length overhead;
- cumulative risk exposure;
- planning and replanning time.

The principal outcome is:

```text
irreversible_failure_rate =
    runs_without_a_feasible_safe_exit / total_runs
```

The central trade-off is the reduction in irreversible failures relative to additional path length and computation.

---

## Repository structure

```text
scenario, map, observations, dynamic obstacle events
        ↓
occupancy, risk, and recoverability representation
        ↓
shortest / risk-aware / recoverability-aware / joint planning
        ↓
route monitoring and online replanning
        ↓
mission outcome, failure reason, metrics, and experiment artifacts
```

The repository includes deterministic planning foundations, risk and recoverability components, incremental replanning experiments, tests, experiment scripts, reports, and an interactive dashboard.

The broader collection of exploratory modules remains available in the [extended research module catalog](docs/CONTRIBUTION_FEATURE_CATALOG.md). Those modules are supporting extensions and are not the primary scientific claim of the project.

---

## Current implementation base

The repository already contains foundations used by the focused research program:

- typed grid, pose, trajectory, and mission primitives;
- deterministic A* and Dijkstra baselines;
- risk-aware grid planning;
- risk and uncertainty fields;
- returnability and recoverability experiments;
- D* Lite incremental replanning;
- repeated obstacle block/clear and moving-start regression tests;
- deterministic experiment entry points and exportable artifacts;
- an interactive Streamlit research laboratory.

The research program is active and implementation-driven. Individual claims are considered established only when they are connected to executable code, tests, reproducible experiments, and stored quantitative results.

---

## Reproducibility requirements

Every reported experiment should provide:

- a documented command;
- deterministic seed support;
- machine-readable configuration;
- CSV or JSON results;
- exact planner parameters;
- baseline comparisons;
- failure-reason reporting;
- dependency and commit information;
- multi-seed aggregate statistics;
- generated tables or figures.

A result is not treated as complete evidence when it is available only as a dashboard visualization or a single demonstration run.

---

## Installation

```bash
git clone https://github.com/panagiotagrosdouli/DynNav.git
cd DynNav
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
```

---

## Run the project

Launch the evidence-bound DynNav Researcher vertical slice:

```bash
python -m pip install -e ".[researcher]"
python -m uvicorn apps.api.main:app --reload --port 8000
```

In a second terminal:

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:3000`. The Researcher compiles a natural-language request into an editable typed protocol, then
runs the existing four-planner Python experiment only after explicit confirmation. Numerical results, statistics, and the
downloadable Markdown report remain unavailable until real execution artifacts exist. See the
[Researcher architecture and roadmap](docs/DYNNAV_RESEARCHER_ARCHITECTURE.md).

The Streamlit laboratory remains available as the legacy research interface:

Launch the interactive laboratory:

```bash
streamlit run app/dashboard.py
```

Run the available benchmark and demonstration entry points:

```bash
dynnav-demo
dynnav-benchmark
python scripts/run_all.py
python scripts/run_benchmarks.py
```

Run the test suite:

```bash
pytest
```

---

## Research roadmap

The focused development sequence is:

1. define and validate the units, ranges, and normalization of every objective term;
2. distinguish expected risk, maximum risk, VaR, and empirical CVaR where used;
3. formalize escape options, returnability, bottleneck exposure, and irreversible failure;
4. integrate recoverability costs with incremental online replanning;
5. build deterministic dynamic bottleneck scenarios;
6. run fair four-way planner ablations;
7. add multi-seed statistics, confidence intervals, and effect sizes;
8. generate publication-quality tables, plots, and failure-case analyses;
9. extend validation to ROS 2, simulation, and physical platforms after the algorithmic and experimental contracts are stable.

The detailed implementation roadmap remains available in [`docs/DYNNAV_V2_RESEARCH_ROADMAP.md`](docs/DYNNAV_V2_RESEARCH_ROADMAP.md).

---

## Evidence boundaries

DynNav is a real research software project, but scientific conclusions must follow the available evidence.

Current repository results are primarily based on deterministic and stochastic synthetic navigation environments. They do not by themselves establish certified safety, universal generalization, production readiness, or physical-robot reliability.

ROS 2/Nav2 integration, simulation-scale validation, physical-robot experiments, and formal safety guarantees require separate executable evidence before they are reported as completed results.

---

## Extended modules

Previous and exploratory work on uncertainty calibration, prediction, supervision, learning, human-aware navigation, multi-robot coordination, security, semantic representations, formal methods, and other extensions is preserved in:

- [Contribution feature catalog](docs/CONTRIBUTION_FEATURE_CATALOG.md)
- [Contribution source index](contributions/CONTRIBUTIONS_README.md)
- [Documentation index](docs/README.md)
- [Interactive dashboard guide](app/README.md)

These components may support future experiments, but the primary DynNav contribution is **risk- and recoverability-aware online replanning that preserves safe escape options under dynamic route invalidation**.

---

## License

DynNav is released under the [Apache License 2.0](LICENSE).
