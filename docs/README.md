# DynNav technical documentation

This directory documents the focused DynNav research program:

> **Risk- and recoverability-aware online replanning under dynamic route invalidation.**

The central question is whether explicit recoverability estimation can reduce irreversible navigation failures without unacceptable path-length or computation overhead.

## Primary research thread

The canonical reading order is:

1. [`RESEARCH_OVERVIEW.md`](RESEARCH_OVERVIEW.md): problem, gap, research question and scope.
2. [`MATHEMATICAL_FORMULATION.md`](MATHEMATICAL_FORMULATION.md): path cost, risk, irreversibility and assumptions.
3. [`RISK_ESTIMATION.md`](RISK_ESTIMATION.md): occupancy-risk definitions and aggregation.
4. [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md): baselines, scenarios, metrics, ablations and fair-comparison rules.
5. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): seeds, commands, configurations and artifact traceability.
6. [`DYNNAV_V2_RESEARCH_ROADMAP.md`](DYNNAV_V2_RESEARCH_ROADMAP.md): implementation and evidence plan.

## Architecture and implementation references

- [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md): package ownership and component boundaries.
- [`NAVIGATION_PIPELINE.md`](NAVIGATION_PIPELINE.md): observation, planning, monitoring and replanning flow.
- [`UNCERTAINTY_MODEL.md`](UNCERTAINTY_MODEL.md): uncertainty representation used by supporting experiments.
- [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md): verified capabilities, evidence gaps and implementation risks.
- [`MARKDOWN_AUDIT.md`](MARKDOWN_AUDIT.md): repository-wide documentation review.
- [`MARKDOWN_STYLE_GUIDE.md`](MARKDOWN_STYLE_GUIDE.md): maturity vocabulary and claim discipline.

## Secondary extensions

The repository also contains exploratory modules in learning, prediction, security, multi-robot coordination, semantic navigation, formal shields and neural scene representations. They are retained as extensions and demonstrations, not as equal parts of the central research claim.

See [`CONTRIBUTION_FEATURE_CATALOG.md`](CONTRIBUTION_FEATURE_CATALOG.md) for the complete catalog.

## Experimental contract

The primary comparison is:

| Variant | Objective |
|---|---|
| shortest | `length` |
| risk-aware | `length + risk` |
| recoverability-aware | `length + irreversibility` |
| combined | `length + risk + irreversibility` |

Every reported result should include:

- exact commit and configuration;
- deterministic seeds;
- scenario and dynamic-obstacle trace;
- metric definitions;
- baseline outputs;
- multi-seed summary;
- limitations and failed cases.

The principal outcome is irreversible failure rate, supported by mission success, recovery success, emergency stops, escape-option count, cumulative risk, path overhead, replans and runtime.

## Verified commands

From the repository root:

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/run_all.py --config configs/default.yaml --smoke --out-dir results/ci_smoke
python scripts/run_benchmarks.py --config configs/default.yaml --smoke --out-dir results/ci_benchmarks
```

Dashboard:

```bash
python -m pip install -e ".[dashboard]"
streamlit run app/dashboard.py
```

## Evidence policy

DynNav is an implementation-driven research project, but implementation is not equivalent to experimental proof. Passing tests establishes consistency with the implemented contract. Synthetic benchmarks establish results only for the evaluated scenarios. Hardware reliability, production ROS 2/Nav2 integration, formal safety and broad generalization require separate evidence.

See the [root README](../README.md), the [Greek README](../README_GR.md), [`configs`](../configs/README.md), [`scripts`](../scripts/README.md) and [`tests`](../tests/README.md).
