# DynNav Presentation Guide

This guide defines a focused, scientifically defensible way to present DynNav in a university setting.

## Core research message

Classical shortest-path planning is insufficient when a mobile robot operates with incomplete maps, noisy sensing, dynamic obstacles, and limited recovery options. DynNav studies how route selection and replanning can explicitly account for risk, uncertainty, recoverability, and mission-level safety actions.

## Recommended presentation claim

> DynNav is a modular research framework for risk-aware dynamic navigation in partially observed environments. It combines deterministic planning baselines with uncertainty, recoverability, route monitoring, replanning, and supervisory safety logic in reproducible synthetic experiments.

Do not present the repository as a certified safety system, a production-ready ROS 2 stack, or validated physical-robot software.

## Focused system pipeline

```text
partially observed scenario
        ↓
occupancy and belief representation
        ↓
uncertainty and risk estimation
        ↓
risk-aware route planning
        ↓
recoverability assessment
        ↓
dynamic route monitoring
        ↓
continue / caution / replan / recover / stop
        ↓
metrics, event log, and exported evidence
```

For a short presentation, focus on uncertainty estimation, risk-aware planning, recoverability, dynamic rerouting, and the supervisor. Present the remaining contribution modules as research extensions.

## Guided presentation application

Install and launch the deterministic presentation mode with:

```bash
python -m pip install -e ".[dev,dashboard]"
streamlit run app/presentation.py
```

The application loads `configs/presentation.yaml` and separates the narrative into four stages:

1. Research problem
2. Baseline comparison
3. Dynamic response
4. Evidence and limitations

The full research playground remains available through:

```bash
streamlit run app/dashboard.py
```

## Canonical live demonstration

Use the versioned scenario in `configs/presentation.yaml`. The demonstration should follow one fixed narrative:

1. Show the map, start, goal, uncertainty field, and initial route.
2. Explain why the shortest route is not necessarily the lowest-risk route.
3. Compare Classical A* and Risk-aware A* under the same environment.
4. Advance the rollout until a dynamic change affects the active route.
5. Show the replanning event and supervisor state.
6. Finish with quantitative metrics and evidence boundaries.

## Experimental comparison

At minimum, compare:

- Dijkstra where supported by the benchmark runner;
- Classical A*;
- Risk-aware A*;
- the full DynNav presentation pipeline.

Report:

- success rate;
- path length;
- expanded nodes;
- runtime;
- number of replans;
- average and maximum route risk;
- recoverability score where implemented;
- supervisor transitions and stop requests.

Use multiple random seeds for reportable results. Report the mean together with dispersion or confidence intervals. A single deterministic run is appropriate for a live demonstration, but it is not sufficient evidence for a general performance claim.

## Automated multi-seed benchmark

Run the presentation benchmark with identical generated environments for both planners:

```bash
python scripts/presentation_benchmark.py \
  --seeds 1-20 \
  --out-dir results/presentation/benchmark
```

The command writes:

- `raw.csv` with one row per planner and seed;
- `aggregate.csv` with mean and population standard deviation;
- `summary.md` with a presentation-ready comparison table;
- `scenario.txt` with the exact scenario parameters used.

For a faster smoke run, use `--seeds 1-3 --max-steps 60`. Runtime measurements are machine-dependent, so only compare planner timings produced on the same machine and in the same run.

## Recommended ablation study

Evaluate the following configurations on the same scenario set:

| Variant | Risk | Uncertainty | Recoverability | Dynamic monitoring | Supervisor |
|---|---:|---:|---:|---:|---:|
| A: Classical A* | No | No | No | No | No |
| B: Risk-aware planning | Yes | No | No | No | No |
| C: Risk and uncertainty | Yes | Yes | No | No | No |
| D: Recovery-aware planning | Yes | Yes | Yes | No | No |
| E: Full DynNav pipeline | Yes | Yes | Yes | Yes | Yes |

The ablation should determine which modules improve safety-related metrics and what computational or path-length cost they introduce.

## Suggested ten-slide structure

1. Navigation problem and motivation.
2. Limitations of shortest-path planning.
3. Research question and scope.
4. DynNav architecture.
5. Risk, uncertainty, and recoverability formulation.
6. Dynamic monitoring and supervisor decisions.
7. Canonical guided demonstration.
8. Experimental protocol and baselines.
9. Results, limitations, and evidence boundaries.
10. Conclusions and next validation steps.

## Scientific evidence boundaries

Clearly distinguish the following categories:

| Category | Appropriate interpretation |
|---|---|
| Implemented and tested | Deterministic software behavior covered by source or regression tests. |
| Research prototype | A working experimental implementation that still requires broader evaluation. |
| Experimental extension | A module intended for controlled exploration rather than a validated subsystem. |
| Documentation concept | A research direction represented mainly through documentation or explanatory demonstrations. |
| Future validation | ROS 2, Gazebo, hardware experiments, and independent safety mechanisms. |

Passing tests supports software consistency. It does not establish real-world safety, generalization, formal guarantees, or hardware reliability.

## Reproducibility checklist

Before presenting or publishing results, record:

- repository commit SHA;
- exact configuration file;
- random seed or seed list;
- command used to run the experiment;
- Python and dependency environment;
- raw result files;
- aggregation script;
- generated table or figure;
- known limitations and failed runs.

## Presentation readiness checklist

- [ ] Run `pytest tests/test_presentation_config.py tests/test_presentation_benchmark.py`.
- [ ] Run a benchmark smoke test with `--seeds 1-3 --max-steps 60`.
- [ ] The guided application opens without import or dependency errors.
- [ ] The canonical scenario produces simulation frames and a valid planner comparison.
- [ ] The live demo has a deterministic fallback recording or screenshots.
- [ ] All reported numbers originate from versioned result files.
- [ ] Baselines use the same maps, seeds, and stopping conditions.
- [ ] Limitations are stated before questions from the audience.
- [ ] Experimental modules are not described as certified capabilities.
- [ ] Screenshots and figures match the commit being presented.
