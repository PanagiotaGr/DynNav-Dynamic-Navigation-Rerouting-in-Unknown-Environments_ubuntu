# DynNav C01–C26 Experiment Catalogue

> Generated from `configs/contributions/registry.yaml` and `configs/contributions/experiments.yaml`.
> All commands below are controlled smoke benchmarks. Passing them is implementation evidence only; it is not real-robot, Gazebo, formal-proof, trained-model, or generalization evidence.

Run the complete dependency-aware suite:

```bash
python scripts/run_contribution_suite.py --output-dir results/contribution_suite
```

The suite emits per-contribution CSV files, `manifest.json`, and `summary.md`. Missing optional dependencies are recorded as explicit skips rather than silently treated as passes.

## C01 — Learned A* Search

- **Maturity:** Research Prototype
- **Hypothesis:** A learned heuristic can reduce A* node expansions without materially reducing success or increasing path length relative to classical A*.
- **Baseline(s):** Classical A*
- **Primary metrics:** `classic_success`, `learned_success`, `classic_expansions`, `learned_expansions`, `expansion_ratio`, `delta_path`, `runtime_ratio`
- **Evidence level:** `synthetic`
- **Optional dependencies:** `torch`
- **Implementation:** [`contributions/01_learned_astar/experiments/eval_astar_learned.py`](../contributions/01_learned_astar/experiments/eval_astar_learned.py)
- **Limitation:** Requires the repository's local trained checkpoint and PyTorch; synthetic grids do not establish transfer to Nav2 or a physical robot.

```bash
python contributions/01_learned_astar/experiments/eval_astar_learned.py --trials 8 --min-goal-distance 8 --out results/contribution_suite/artifacts/c01_learned_astar.csv
```

## C02 — Uncertainty Calibration

- **Maturity:** Research Prototype
- **Hypothesis:** Post-hoc calibration improves absolute-error calibration and empirical interval coverage relative to raw uncertainty estimates.
- **Baseline(s):** Raw uncertainty, Affine calibration
- **Primary metrics:** `ece_abs_error`, `coverage_1sigma`, `coverage_2sigma`, `pearson_uncertainty_error`, `spearman_uncertainty_error`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/02_uncertainty_calibration/experiments/eval_uncertainty_calibration.py`](../contributions/02_uncertainty_calibration/experiments/eval_uncertainty_calibration.py)
- **Limitation:** The default generator is synthetic and cannot establish calibration under sensor drift or domain shift.

```bash
python contributions/02_uncertainty_calibration/experiments/eval_uncertainty_calibration.py --n 200 --seed 13 --out results/contribution_suite/artifacts/c02_uncertainty_calibration.csv
```

## C03 — Belief and Risk Planning

- **Maturity:** Research Prototype
- **Hypothesis:** CVaR-aware route selection reduces tail occupancy risk with a measurable path-length trade-off relative to risk-neutral routing.
- **Baseline(s):** Risk-neutral route, Expected-risk route
- **Primary metrics:** `path_length`, `expected_risk`, `cvar_risk`, `max_risk`, `total_objective`, `dominated`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/03_belief_risk_planning/experiments/eval_risk_tradeoff.py`](../contributions/03_belief_risk_planning/experiments/eval_risk_tradeoff.py)
- **Limitation:** Route candidates and beliefs are controlled fixtures rather than live occupancy estimates.

```bash
python contributions/03_belief_risk_planning/experiments/eval_risk_tradeoff.py --alpha 0.95 --objective-risk cvar --out results/contribution_suite/artifacts/c03_belief_risk_tradeoff.csv
```

## C04 — Irreversibility and Returnability

- **Maturity:** Research Prototype
- **Hypothesis:** Recoverability loss and bottleneck exposure distinguish fragile paths that conventional path length alone treats as acceptable.
- **Baseline(s):** Geometric path length
- **Primary metrics:** `min_recoverability`, `cumulative_recoverability_loss`, `bottleneck_exposure`, `max_irreversibility`, `all_returnable`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/04_irreversibility_returnability/experiments/eval_recoverability_metrics.py`](../contributions/04_irreversibility_returnability/experiments/eval_recoverability_metrics.py)
- **Limitation:** Recoverability is evaluated on discrete graph fixtures and is not yet a calibrated real-world failure probability.

```bash
python contributions/04_irreversibility_returnability/experiments/eval_recoverability_metrics.py --out results/contribution_suite/artifacts/c04_recoverability_profile.csv
```

## C05 — Safe-Mode Navigation

- **Maturity:** Research Prototype
- **Hypothesis:** Hysteresis and persistence thresholds reduce mode chattering while escalating high-risk traces to safe mode or emergency stop.
- **Baseline(s):** Single-threshold supervisor
- **Primary metrics:** `safe_mode_steps`, `emergency_stop_steps`, `transitions`, `replans`, `operator_alerts`, `mean_commanded_speed`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/05_safe_mode_navigation/experiments/eval_safe_mode_thresholds.py`](../contributions/05_safe_mode_navigation/experiments/eval_safe_mode_thresholds.py)
- **Limitation:** Controller outputs are policy decisions on scripted traces, not closed-loop velocity validation.

```bash
python contributions/05_safe_mode_navigation/experiments/eval_safe_mode_thresholds.py --n-steps 20 --out results/contribution_suite/artifacts/c05_safe_mode_thresholds.csv
```

## C06 — Energy and Connectivity

- **Maturity:** Research Prototype
- **Hypothesis:** Explicit energy reserve and connectivity constraints reject mission routes that distance-only planning would incorrectly accept.
- **Baseline(s):** Distance-only feasibility
- **Primary metrics:** `energy_margin`, `connectivity_margin`, `feasible`, `selected_best`, `via_recharge`, `via_relay`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/06_energy_connectivity/experiments/eval_resource_feasibility.py`](../contributions/06_energy_connectivity/experiments/eval_resource_feasibility.py)
- **Limitation:** Energy and radio models are analytical proxies without hardware characterization.

```bash
python contributions/06_energy_connectivity/experiments/eval_resource_feasibility.py --battery-budget 26 --reserve-energy 2 --min-connectivity 0.35 --out results/contribution_suite/artifacts/c06_resource_feasibility.csv
```

## C07 — Next-Best View

- **Maturity:** Research Prototype
- **Hypothesis:** Returnability-aware next-best-view scoring preserves safer retreat options than information-gain-only selection.
- **Baseline(s):** Information-gain-only NBV
- **Primary metrics:** `information_gain`, `path_risk`, `returnability`, `connectivity`, `classic_score`, `safe_score`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/07_next_best_view/experiments/eval_returnability_aware_nbv.py`](../contributions/07_next_best_view/experiments/eval_returnability_aware_nbv.py)
- **Limitation:** Candidate views are hand-authored fixtures rather than sensor-derived frontiers.

```bash
python contributions/07_next_best_view/experiments/eval_returnability_aware_nbv.py --risk-weight 1.0 --returnability-weight 1.0 --connectivity-weight 0.25 --out results/contribution_suite/artifacts/c07_returnability_aware_nbv.csv
```

## C08 — Security IDS

- **Maturity:** Experimental
- **Hypothesis:** Severity- and trust-conditioned IDS response produces graded mitigations rather than a brittle binary alarm policy.
- **Baseline(s):** Binary alarm-only response
- **Primary metrics:** `flag_rate`, `triggered`, `severity`, `mitigation`, `trust_score`, `d2_ratio`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/08_security_ids/experiments/eval_ids_response_policy.py`](../contributions/08_security_ids/experiments/eval_ids_response_policy.py)
- **Limitation:** Synthetic detector scores do not demonstrate detection performance against real ROS 2 attacks.

```bash
python contributions/08_security_ids/experiments/eval_ids_response_policy.py --out results/contribution_suite/artifacts/c08_ids_response_policy.csv
```

## C09 — Multi-Robot Navigation

- **Maturity:** Experimental
- **Hypothesis:** Joint conflict and risk-budget accounting identifies unsafe multi-robot plans missed by independent single-robot validation.
- **Baseline(s):** Independent robot plans
- **Primary metrics:** `n_conflicts`, `vertex_conflicts`, `edge_swap_conflicts`, `risk_budget_violations`, `belief_disagreement_count`, `feasible`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/09_multi_robot/experiments/eval_team_coordination.py`](../contributions/09_multi_robot/experiments/eval_team_coordination.py)
- **Limitation:** The benchmark audits predefined plans and does not exercise distributed ROS 2 communication.

```bash
python contributions/09_multi_robot/experiments/eval_team_coordination.py --out results/contribution_suite/artifacts/c09_team_coordination.csv
```

## C10 — Human, Language, and Ethics

- **Maturity:** Experimental
- **Hypothesis:** Human distance, operator trust, zone semantics, and instruction confidence should produce auditable speed and autonomy restrictions.
- **Baseline(s):** Context-free navigation policy
- **Primary metrics:** `action`, `max_speed`, `autonomy_level`, `path_allowed`, `requires_operator_confirmation`, `ethical_cost_multiplier`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/10_human_language_ethics/experiments/eval_human_ethics_policy.py`](../contributions/10_human_language_ethics/experiments/eval_human_ethics_policy.py)
- **Limitation:** Policy fixtures do not constitute human-subject validation or normative proof.

```bash
python contributions/10_human_language_ethics/experiments/eval_human_ethics_policy.py --out results/contribution_suite/artifacts/c10_human_ethics_policy.csv
```

## C11 — VLM Navigation Agent

- **Maturity:** Experimental
- **Hypothesis:** A deterministic safety validator rejects low-confidence, malformed, or out-of-bounds VLM goals before metric planning.
- **Baseline(s):** Unvalidated VLM output
- **Primary metrics:** `decision`, `valid_confidence`, `valid_region`, `valid_pixel`, `valid_waypoint`, `reason`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/11_vlm_navigation_agent/experiments/eval_vlm_goal_validation.py`](../contributions/11_vlm_navigation_agent/experiments/eval_vlm_goal_validation.py)
- **Limitation:** This validates structured outputs only; it does not call or benchmark a real vision-language model.

```bash
python contributions/11_vlm_navigation_agent/experiments/eval_vlm_goal_validation.py --out results/contribution_suite/artifacts/c11_vlm_goal_validation.csv
```

## C12 — Diffusion Occupancy

- **Maturity:** Experimental
- **Hypothesis:** Probabilistic occupancy forecasts improve calibration and high-risk recall relative to deterministic persistence baselines.
- **Baseline(s):** Deterministic persistence, Frequency forecast
- **Primary metrics:** `brier_score`, `nll`, `high_risk_precision`, `high_risk_recall`, `cvar_conservatism_gap`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/12_diffusion_occupancy/experiments/eval_risk_map_quality.py`](../contributions/12_diffusion_occupancy/experiments/eval_risk_map_quality.py)
- **Limitation:** The lightweight predictor is a controlled proxy and not evidence for a trained diffusion model.

```bash
python contributions/12_diffusion_occupancy/experiments/eval_risk_map_quality.py --n-scenarios 4 --out results/contribution_suite/artifacts/c12_risk_map_quality.csv
```

## C13 — Latent World Model

- **Maturity:** Experimental
- **Hypothesis:** Latent rollout auditing selects action sequences that balance predicted return with effort and recoverability penalties.
- **Baseline(s):** Return-only rollout ranking
- **Primary metrics:** `imagined_return`, `action_effort`, `terminal_latent_norm`, `recoverability_proxy`, `irreversible`, `final_score`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/13_latent_world_model/experiments/eval_rollout_audit.py`](../contributions/13_latent_world_model/experiments/eval_rollout_audit.py)
- **Limitation:** The latent dynamics are untrained fixtures and cannot support world-model accuracy claims.

```bash
python contributions/13_latent_world_model/experiments/eval_rollout_audit.py --horizon 6 --out results/contribution_suite/artifacts/c13_rollout_audit.csv
```

## C14 — Causal Risk Attribution

- **Maturity:** Experimental
- **Hypothesis:** Counterfactual intervention ranking recovers the injected navigation failure cause more reliably than outcome magnitude alone.
- **Baseline(s):** Outcome-magnitude ranking
- **Primary metrics:** `top1_correct`, `true_cause_rank`, `counterfactual_reduction`, `predicted_root_cause`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/14_causal_risk_attribution/experiments/eval_root_cause_attribution.py`](../contributions/14_causal_risk_attribution/experiments/eval_root_cause_attribution.py)
- **Limitation:** Correctness is relative to a hand-specified structural causal model.

```bash
python contributions/14_causal_risk_attribution/experiments/eval_root_cause_attribution.py --n-samples 30 --out results/contribution_suite/artifacts/c14_root_cause_attribution.csv
```

## C15 — Neuromorphic Sensing

- **Maturity:** Experimental
- **Hypothesis:** Event-stream obstacle detection can reduce detection latency for moving targets while maintaining low false-negative rates relative to frame sampling.
- **Baseline(s):** Frame-based detector
- **Primary metrics:** `detected`, `detection_time_us`, `latency_us`, `false_negative`, `event_rate_per_ms`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/15_neuromorphic_sensing/experiments/eval_neuromorphic_latency.py`](../contributions/15_neuromorphic_sensing/experiments/eval_neuromorphic_latency.py)
- **Limitation:** Simulated events and Python timing do not establish hardware event-camera latency.

```bash
python contributions/15_neuromorphic_sensing/experiments/eval_neuromorphic_latency.py --out results/contribution_suite/artifacts/c15_neuromorphic_latency.csv
```

## C16 — Federated Navigation Learning

- **Maturity:** Experimental
- **Hypothesis:** Robust or privacy-aware aggregation exposes measurable accuracy, fairness, privacy, and communication trade-offs across robot clients.
- **Baseline(s):** Centralized reference, FedAvg
- **Primary metrics:** `mean_client_mse`, `worst_client_mse`, `fairness_gap`, `communication_floats`, `final_server_val_mse`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/16_federated_nav_learning/experiments/eval_federated_tradeoffs.py`](../contributions/16_federated_nav_learning/experiments/eval_federated_tradeoffs.py)
- **Limitation:** Synthetic client objectives do not establish performance on heterogeneous robot datasets.

```bash
python contributions/16_federated_nav_learning/experiments/eval_federated_tradeoffs.py --rounds 3 --n-clients 4 --out results/contribution_suite/artifacts/c16_federated_tradeoffs.csv
```

## C17 — Topological Semantic Maps

- **Maturity:** Experimental
- **Hypothesis:** Sparse semantic grounding and edge invalidation support correct goal retrieval and route adaptation on a topological map.
- **Baseline(s):** Nominal topological route
- **Primary metrics:** `top1_correct`, `topk_correct`, `path_found`, `path_cost`, `blocked_edges`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/17_topological_semantic_maps/experiments/eval_topo_semantic_navigation.py`](../contributions/17_topological_semantic_maps/experiments/eval_topo_semantic_navigation.py)
- **Limitation:** Hash-based label embeddings are deterministic stubs, not open-vocabulary perception evidence.

```bash
python contributions/17_topological_semantic_maps/experiments/eval_topo_semantic_navigation.py --out results/contribution_suite/artifacts/c17_topological_semantic_navigation.csv
```

## C18 — Formal Safety Shields

- **Maturity:** Experimental
- **Hypothesis:** Runtime shielding reduces safety violations and minimum-distance breaches relative to unshielded control under matched scenarios.
- **Baseline(s):** Unshielded controller
- **Primary metrics:** `safety_violations`, `min_obstacle_distance`, `min_stl_robustness`, `intervention_rate`, `final_goal_distance`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/18_formal_safety_shields/experiments/eval_shield_stress_test.py`](../contributions/18_formal_safety_shields/experiments/eval_shield_stress_test.py)
- **Limitation:** Numerical stress tests are not a formal proof and do not cover full robot dynamics.

```bash
python contributions/18_formal_safety_shields/experiments/eval_shield_stress_test.py --out results/contribution_suite/artifacts/c18_shield_stress_test.csv
```

## C19 — LLM Mission Planner

- **Maturity:** Documentation Concept
- **Hypothesis:** Schema-constrained mission planning improves waypoint order, constraint satisfaction, and execution readiness over unchecked language output.
- **Baseline(s):** Unchecked mission sequence
- **Primary metrics:** `ordering_accuracy`, `exact_sequence_match`, `unresolved_waypoints`, `forbidden_violations`, `execution_ready`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/19_llm_mission_planner/experiments/eval_mission_plan_quality.py`](../contributions/19_llm_mission_planner/experiments/eval_mission_plan_quality.py)
- **Limitation:** The benchmark uses deterministic fixtures and does not establish LLM reliability.

```bash
python contributions/19_llm_mission_planner/experiments/eval_mission_plan_quality.py --out results/contribution_suite/artifacts/c19_mission_plan_quality.csv
```

## C20 — Multimodal Failure Explainer

- **Maturity:** Experimental
- **Hypothesis:** Structured multimodal failure reports increase root-cause coverage and operator-action relevance over minimal event summaries.
- **Baseline(s):** Minimal event summary
- **Primary metrics:** `completeness_score`, `root_cause_recall`, `action_relevance`, `stl_coverage`, `operator_readiness`, `total_score`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/20_multimodal_failure_explainer/experiments/eval_failure_report_quality.py`](../contributions/20_multimodal_failure_explainer/experiments/eval_failure_report_quality.py)
- **Limitation:** Automated rubric scores are not a substitute for blinded operator studies.

```bash
python contributions/20_multimodal_failure_explainer/experiments/eval_failure_report_quality.py --out results/contribution_suite/artifacts/c20_failure_report_quality.csv
```

## C21 — PPO Navigation Agent

- **Maturity:** Experimental
- **Hypothesis:** Policy-level safety evaluation reveals reward-safety trade-offs that aggregate return alone would conceal.
- **Baseline(s):** Unshielded policy, Heuristic policy
- **Primary metrics:** `success_rate`, `collision_rate`, `mean_reward`, `mean_min_obstacle_distance`, `shield_intervention_rate`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/21_ppo_navigation_agent/experiments/eval_policy_safety.py`](../contributions/21_ppo_navigation_agent/experiments/eval_policy_safety.py)
- **Limitation:** The policies and environment are lightweight fixtures, not trained PPO deployment evidence.

```bash
python contributions/21_ppo_navigation_agent/experiments/eval_policy_safety.py --episodes 8 --out results/contribution_suite/artifacts/c21_policy_safety.csv
```

## C22 — Curriculum RL

- **Maturity:** Experimental
- **Hypothesis:** Adaptive curriculum schedules reach hard scenarios with better stability and sample efficiency than fixed difficulty schedules.
- **Baseline(s):** Fixed curriculum, Linear curriculum
- **Primary metrics:** `episodes_to_hard`, `success_trend`, `stability_score`, `heldout_transfer_success`, `sample_efficiency_score`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/22_curriculum_rl/experiments/eval_curriculum_strategies.py`](../contributions/22_curriculum_rl/experiments/eval_curriculum_strategies.py)
- **Limitation:** Synthetic learning curves do not demonstrate sample efficiency for a trained navigation agent.

```bash
python contributions/22_curriculum_rl/experiments/eval_curriculum_strategies.py --episodes 40 --out results/contribution_suite/artifacts/c22_curriculum_strategies.csv
```

## C23 — Gaussian Splatting Mapper

- **Maturity:** Documentation Concept
- **Hypothesis:** Incremental Gaussian-map proxies expose occupancy quality, uncertainty separation, and frontier utility as observations accumulate.
- **Baseline(s):** Sparse occupancy proxy
- **Primary metrics:** `occupancy_iou`, `occupancy_precision`, `occupancy_recall`, `uncertainty_unknown_gap`, `frontier_precision_proxy`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/23_gaussian_splatting_mapper/experiments/eval_gs_mapping_quality.py`](../contributions/23_gaussian_splatting_mapper/experiments/eval_gs_mapping_quality.py)
- **Limitation:** This is a 2D Gaussian proxy, not photorealistic 3D Gaussian Splatting reconstruction.

```bash
python contributions/23_gaussian_splatting_mapper/experiments/eval_gs_mapping_quality.py --frames 4 --out results/contribution_suite/artifacts/c23_gaussian_mapping_quality.csv
```

## C24 — NeRF Uncertainty

- **Maturity:** Documentation Concept
- **Hypothesis:** Rendering-uncertainty proxies separate novel views from known space and improve risk-aware exploration decisions.
- **Baseline(s):** Uniform uncertainty, Uncertainty-agnostic planning
- **Primary metrics:** `brier_score`, `ece`, `ood_auroc`, `novel_view_uncertainty_gap`, `exploration_precision_at_k`, `planning_safety_gain`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/24_nerf_uncertainty/experiments/eval_nerf_uncertainty.py`](../contributions/24_nerf_uncertainty/experiments/eval_nerf_uncertainty.py)
- **Limitation:** The evaluator uses synthetic uncertainty fields and does not train or render a NeRF.

```bash
python contributions/24_nerf_uncertainty/experiments/eval_nerf_uncertainty.py --out results/contribution_suite/artifacts/c24_nerf_uncertainty.csv
```

## C25 — Adversarial Attack Simulator

- **Maturity:** Experimental
- **Hypothesis:** Attack-specific impact scoring distinguishes geometry, odometry, and sensor degradation and selects proportionate mitigations.
- **Baseline(s):** No-attack scenario
- **Primary metrics:** `detected`, `severity_score`, `min_distance_degradation`, `geometry_change`, `odometry_error_m`, `mitigation`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/25_adversarial_attack_simulator/experiments/eval_attack_impact.py`](../contributions/25_adversarial_attack_simulator/experiments/eval_attack_impact.py)
- **Limitation:** Injected numerical attacks are not an end-to-end ROS 2 penetration test.

```bash
python contributions/25_adversarial_attack_simulator/experiments/eval_attack_impact.py --out results/contribution_suite/artifacts/c25_attack_impact.csv
```

## C26 — Swarm Consensus

- **Maturity:** Experimental
- **Hypothesis:** Trust-aware robust consensus maintains agreement and mission success under Byzantine participants better than unweighted aggregation.
- **Baseline(s):** Mean consensus, Median consensus
- **Primary metrics:** `consensus_accuracy`, `mission_success`, `byzantine_detection_recall`, `communication_messages`, `trust_weighted_agreement`
- **Evidence level:** `synthetic`
- **Optional dependencies:** None
- **Implementation:** [`contributions/26_swarm_consensus/experiments/eval_swarm_consensus.py`](../contributions/26_swarm_consensus/experiments/eval_swarm_consensus.py)
- **Limitation:** In-process simulation does not model real network delays, packet ordering, or ROS 2 DDS failure modes.

```bash
python contributions/26_swarm_consensus/experiments/eval_swarm_consensus.py --out results/contribution_suite/artifacts/c26_swarm_consensus.csv
```
