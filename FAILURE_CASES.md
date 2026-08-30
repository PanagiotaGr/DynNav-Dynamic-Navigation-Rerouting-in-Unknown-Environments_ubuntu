# Failure Cases and Falsification Suite

| Case | Construction | Expected winner / falsification signal |
|---|---|---|
| Open static room | equal-risk open alternatives, no event | J0/NavFn should minimize length and latency; J2 overhead is unnecessary |
| Hazardous wide route | shortest corridor has calibrated high traversal risk but retreat remains easy | J1 should dominate J2 |
| Matched-risk commitment | equal risk/length within tolerance; one route loses safe-region reachability after randomized closure | J2 should reduce recovery-infeasible failure |
| False conservatism | only feasible corridor is narrow and never closes | J2 rejection, timeout, or detour is harm |
| Wrong local degree | cul-de-sac with many shallow side pockets | local heuristic predicts high options although safe-region reachability is poor |
| Single narrow bridge to open basin | bottleneck must be crossed to reach a large recoverable space | accumulated local penalty may reject the globally correct decision |
| Risk–recovery conflict | low-risk corridor has weak retreat; higher-risk route preserves retreat | J3 must expose a stable, preregistered trade-off rather than cherry-picked weights |
| Event off selected route | randomized closure affects neither or the alternative route | recovery-aware planner must not receive automatic credit |
| Late observation | event occurs before perception, after physical commitment | tests whether estimates based on stale costmaps are misleading |
| Retreat blocked behind robot | closure removes return path after commitment | intended positive case; failure to retreat falsifies usefulness |
| Forward route blocked, retreat open | goal route closes but safe region remains reachable | mission failure must not be mislabeled irreversible |
| Moving obstacle clears | temporary obstruction | excessive rerouting/recovery reveals hysteresis or conservatism |
| Oscillating block/clear | periodic live obstacle | pathological replanning, route oscillation, and latency are measured |
| Inflation sensitivity | same geometry across footprint/inflation radii | rank reversal indicates metric is configuration-dependent |
| Resolution sensitivity | same metric world at multiple costmap resolutions | local degree should not change scientific conclusion solely due to discretization |
| Orientation/kinodynamics | turn-constrained robot in geometrically open cells | grid recoverability may claim escape when controller cannot execute it |
| Localization drift | apparent return corridor differs from truth | separate robot-belief prediction from ground-truth outcome |

Every case is retained whether it helps or harms DynNav. A scenario is excluded from efficacy only for a preregistered invalidity reason, never because a planner performed poorly.
