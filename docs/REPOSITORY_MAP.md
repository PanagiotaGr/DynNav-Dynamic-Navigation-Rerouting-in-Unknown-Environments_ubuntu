# DynNav repository map

This page explains where to look in the repository and, more importantly, why each area exists.

If you are new to robotics, start with the root `README.md` and `docs/PROJECT_OVERVIEW.md` before reading source code.

## The project in four layers

DynNav is easier to understand if the repository is treated as four connected layers rather than as a flat collection of folders.

### 1. Core research algorithm

`dynnav/` is the canonical Python research package. It contains the reference planning algorithms, environment representation, risk/recoverability logic, evaluation code and research protocol support.

This is the best place to inspect the algorithm independently of ROS.

### 2. Robot integration and simulation

`ros2_ws/src/` contains the ROS 2 side of the project.

- `dynnav_nav2_cpp/` — the C++ Nav2 global-planner plugin.
- `dynnav_nav2_benchmark/` — benchmark and dynamic route-invalidation experiments.
- `dynnav_turtlebot3/` — TurtleBot3 simulation/bring-up material.

These packages answer a different question from the Python package: can the research idea be integrated into the navigation stack used by a robot?

### 3. Experiments and evidence

`configs/` defines experiment parameters. `scripts/` contains reproducible runners. `tests/` checks software and research contracts. `results/` stores retained outputs and evidence. `docs/` explains protocols, interpretation and limitations.

The intended flow is:

```text
configuration
    ↓
experiment runner
    ↓
planner / simulator
    ↓
raw trial data
    ↓
metrics and statistical analysis
    ↓
report / retained evidence
```

### 4. Human-facing tools and exploratory work

`app/` is the Streamlit laboratory. `apps/api/` and `apps/web/` form the Researcher interface. These make experiments easier to inspect and run; they are not the scientific contribution themselves.

`contributions/` contains C01–C26 exploratory modules. They represent broader research directions and prototypes. They should not be confused with the smallest publishable DynNav contribution, which is defined in `CORE_CONTRIBUTION.md`.

## Directory guide

| Path | Role | Start here when... |
|---|---|---|
| `dynnav/` | canonical Python research implementation | you want to understand the algorithm |
| `ros2_ws/src/` | ROS 2/Nav2 integration | you want to understand robot/simulation integration |
| `configs/` | experiment configuration | you want to reproduce a run |
| `scripts/` | experiment and validation commands | you want to execute the project |
| `tests/` | regression/research-contract tests | you want to verify implementation behaviour |
| `results/` | retained evidence | you want to inspect outputs rather than claims |
| `docs/` | protocols, explanations and research status | you want scientific context |
| `paper/` | manuscript-facing material | you want the publication path |
| `app/` | Streamlit research laboratory | you want an interactive local interface |
| `apps/` | Researcher API/web workspace | you want the richer research interface |
| `contributions/` | exploratory C01–C26 work | you want broader ideas beyond the core paper |
| `analysis/` | analysis utilities | you want post-processing and investigation code |
| `datasets/`, `data/`, `data_curriculum/` | retained/experimental data areas | you are tracing a data-dependent experiment |
| `assets/`, `figures/` | visual material | you are looking for diagrams/plots |

## Important root documents

The root intentionally contains a small number of research-governance files:

- `README.md` — entry point for humans.
- `CORE_CONTRIBUTION.md` — smallest scientifically defensible contribution.
- `EXPERIMENT_PROTOCOL_V2.md` — normative efficacy experiment protocol.
- `CLAIM_EVIDENCE_MATRIX.md` — maps claims to evidence.
- `REPRODUCIBILITY_REPORT.md` — reproducibility status.
- `FAILURE_CASES.md` — known failure modes.
- `PUBLICATION_PLAN.md` — publication path.
- `STATUS.yaml` — machine-readable status.

## How to review the project

For a quick review, read:

```text
README.md
→ docs/PROJECT_OVERVIEW.md
→ CORE_CONTRIBUTION.md
→ CLAIM_EVIDENCE_MATRIX.md
```

For a technical review, continue with:

```text
dynnav/
→ tests/
→ configs/
→ scripts/
→ results/
```

For a robotics review, continue with:

```text
ros2_ws/src/dynnav_nav2_cpp/
→ ros2_ws/src/dynnav_nav2_benchmark/
→ docs/GAZEBO_BENCHMARK_PROTOCOL.md
→ docs/DYNAMIC_EXECUTION_PROTOCOL.md
```

The guiding rule is simple: source code shows what is implemented, tests show software contracts, retained experiment artifacts show what was observed, and documentation states only conclusions supported by those layers.
