# Smallest Publishable Core

## One-sentence contribution

DynNav tests whether an explicitly measured, robot-information-conditioned estimate of post-invalidation recovery feasibility improves online replanning outcomes over geometric and costmap-risk objectives in controlled Nav2/Gazebo navigation.

Until such an estimator replaces or complements the current local heuristic, the narrower honest wording is: **a controlled evaluation of local escape-option regularization in Nav2 global planning**.

## Frozen comparison

| ID | Objective | Purpose |
|---|---|---|
| J0 | \(L\) | shortest-only control |
| J1 | \(L+\lambda_R R\) | costmap-risk control |
| J2 | \(L+\lambda_Q(1-Q)\) | recoverability treatment |
| J3 | \(L+\lambda_R R+\lambda_Q(1-Q)\) | joint treatment |

NavFn and SmacPlanner2D are external engineering references. The causal scientific ablation is J0–J3 implemented in the same plugin and ROS stack.

## Operational quantities

* `mission_success`: goal reached within `T_mission`, without collision or emergency stop.
* `valid_invalidation`: event service succeeds, obstacle is visible in the relevant costmap by `T_observe`, and the pre-event path intersects the newly lethal footprint.
* `replan`: a new planner-server result, after observation, whose path identifier/timestamp differs from the pre-event plan.
* `recovery_feasible`: from the first valid post-event state, an allowed recovery execution reaches the preregistered safe region within `T_recovery` and `D_recovery`, collision-free.
* `irreversible_failure`: valid invalidation, mission failure, and `recovery_feasible=false`. Report the less metaphysical label “post-invalidation recovery-infeasible failure” in tables.
* `recovery_success`: mission is abandoned or blocked, but the executed recovery policy reaches the safe region within budget.
* `executed_path_length`: integral of ground-truth planar displacement after removing teleport/reset discontinuities; odometry length is a secondary deployable estimate.
* `planning_latency`: planner-server request acceptance to result, measured per initial plan and replan; report warm-up separately.
* `number_of_replans`: count of completed post-initial planner-server requests for the active goal.
* `cumulative_risk`: line integral of the frozen robot-visible risk field sampled along executed trajectory; it is a score unless calibrated.
* `escape_option_preservation`: minimum and area-under-time of `Q_t` along the pre-event executed trajectory. State whether `Q` is the local heuristic or validated estimator.

## Scope exclusions

No learned heuristics, multi-robot, security, VLA/LLM, NeRF, federated learning, or dashboard feature is needed for the paper. No formal safety, probability, hardware, or universal superiority claim is in scope. The current 26-contribution catalogue should be presented as exploratory work, not evidence for this paper.

## Go/no-go criterion

Proceed to a paper only if held-out dynamic trials contain enough discordant irreversible outcomes to estimate a nontrivial paired effect and the upper 95% CI for overhead stays below frozen margins. A null or harmful result is publishable only if the benchmark and estimator validation are strong and failure mechanisms are analyzed.
