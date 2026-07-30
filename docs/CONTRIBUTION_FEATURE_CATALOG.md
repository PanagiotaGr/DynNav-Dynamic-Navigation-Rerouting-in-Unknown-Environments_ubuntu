# DynNav extension catalog

DynNav now has one primary research contribution:

> **Risk- and recoverability-aware online replanning under dynamic route invalidation.**

The repository also contains a broad set of earlier research modules. They remain available for reuse, comparison and future extensions, but they are not equal parts of the central research claim.

## Core modules

The focused research program directly builds on:

| ID | Module | Role in the main research program |
|---|---|---|
| **C03** | Risk-Aware A* | Supplies occupancy-risk route costs and risk-only ablations. |
| **C04** | Returnability and Recoverability | Supplies escape-option, returnability, bottleneck and irreversibility concepts. |
| **C05** | Safe-Mode Supervisor | Supports explicit replan, recover and stop responses after risk or recoverability degradation. |

Classical A*, Dijkstra and D* Lite implementations provide the geometric and online-replanning baselines.

## Supporting modules

These modules may support later experiments without expanding the main claim:

| ID | Module | Possible supporting role |
|---|---|---|
| **C02** | Uncertainty Estimation | Test sensitivity to uncertain or stale occupancy information. |
| **C07** | Safe Next-Best View | Study whether exploration targets preserve safe return options. |
| **C12** | Diffusion Occupancy Prediction | Future comparison against learned dynamic-occupancy prediction. |
| **C14** | Causal Risk Attribution | Explain whether failure originated from risk, bottleneck exposure or return loss. |
| **C18** | Formal Safety Shields | Future runtime layer after planner evaluation is stable. |
| **C20** | Failure Explanation | Generate structured explanations from event traces. |
| **C25** | Adversarial Navigation Testing | Stress-test route invalidation and observation perturbations. |

## Exploratory extensions

The following modules remain in the repository as independent exploratory directions:

- **C01** — Learned A* Search
- **C06** — Energy and Connectivity
- **C08** — Security and Intrusion Detection
- **C09** — Multi-Robot Coordination
- **C10** — Human-Aware Navigation
- **C11** — Twin-Critic Reinforcement Learning
- **C13** — Latent World Model
- **C15** — Neuromorphic Sensing
- **C16** — Federated Navigation Learning
- **C17** — Semantic Topological Maps
- **C19** — Language Mission Planner
- **C21** — PPO Navigation
- **C22** — Curriculum Reinforcement Learning
- **C23** — Gaussian Splatting Maps
- **C24** — NeRF Uncertainty
- **C26** — Byzantine-Fault-Tolerant Swarm

These extensions should not delay the focused work on objective contracts, recoverability metrics, deterministic route-invalidation scenarios, ablations and multi-seed evaluation.

## Interactive access

The existing dashboard remains available as an inspection and demonstration interface:

```bash
python -m pip install -e ".[dashboard]"
streamlit run app/dashboard.py
```

Open **Contribution Explorer** to inspect the individual modules. The canonical dashboard metadata is stored in [`src/dynnav_dashboard/contribution_registry.yaml`](../src/dynnav_dashboard/contribution_registry.yaml).

## Detailed legacy documentation

For module-level source code, experiments, figures and bilingual documentation, see:

- [`contributions/CONTRIBUTIONS_README.md`](../contributions/CONTRIBUTIONS_README.md)
- [`contributions/`](../contributions/)

## Evidence interpretation

A renderer, figure, test or synthetic benchmark does not by itself establish real-robot safety, broad generalization, formal correctness, ROS 2 integration or production readiness. Each module retains its own maturity and evidence boundary.
