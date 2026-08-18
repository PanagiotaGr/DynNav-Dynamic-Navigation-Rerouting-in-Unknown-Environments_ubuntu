# Claim–Evidence Matrix

| Claim | Required evidence | Current evidence | Status | Missing experiment |
|---|---|---|---|---|
| J0–J3 objectives are implemented | deterministic known-answer tests and zero-weight equivalence | C++ grid search and unit tests; static six-configuration artifact | SUPPORTED | none for implementation claim |
| DynNav loads as a Nav2 plugin | Jazzy build, pluginlib discovery, planner-server activation logs | CI definition and plugin-load test; retained planner-server results, no raw lifecycle logs | PARTIALLY SUPPORTED | retain configure/activate/createPlan/deactivate logs from clean run |
| DynNav returns paths through planner_server | action result and planner-server trace | retained 36/36 static `ComputePathToPose` artifact | SUPPORTED | repeat at paper revision with non-null SHA and logs |
| Dynamic obstacle is perceived at runtime | Gazebo event plus post-event live costmap evidence | 8/8 commissioning trials show required lethal-cell increase and costmap snapshots | SUPPORTED | repeat across final scenarios/seeds |
| Online replanning occurred | timestamped pre/post planner-server calls and paths | direct `/plan` capture and event-relative replan counting implemented but not yet executed | PARTIALLY SUPPORTED | execute Jazzy/Gazebo V2 and retain paths/logs |
| Recoverability metric estimates recovery feasibility | held-out discrimination/calibration against executed recovery outcomes | local neighbor-count heuristic and synthetic route profiles | UNSUPPORTED | estimator validation on held-out rollout labels |
| H1: recoverability reduces irreversible failures | powered paired dynamic trials with failures and CIs | bias-controlled Level 1 V2 synthetic run exists; dynamic `n=1` still has 0/8 irreversible failures | UNSUPPORTED | execute powered Level 4 V2 benchmark |
| H2: benefit increases with uncertainty | factorial uncertainty interaction | no adequate dynamic uncertainty sweep | UNSUPPORTED | frozen multi-level uncertainty experiment |
| H3: overhead is bounded | preregistered margins and CIs on paired dynamic trials | static path/latency descriptive data only | UNSUPPORTED | dynamic non-inferiority analysis |
| H4: joint outperforms components where appropriate | balanced risk/recovery mechanism families and J0–J3 | one joint timeout; no J2/J0 dynamic trials | UNSUPPORTED | V2 aligned/conflict/neutral families |
| Static planner configurations succeed on retained queries | raw paths/results, parameters, map, versions | 36/36 retained requests | SUPPORTED | avoid generalizing beyond two queries |
| Dynamic Gazebo navigation works | robot execution with Nav2 and event | retained 8-trial commissioning artifact, 7 successes | SUPPORTED | evidence is commissioning, not efficacy |
| Operational irreversibility is measurable | validated recovery label and event-time state | graph reachability on post-event inflated costmap | PARTIALLY SUPPORTED | compare graph label to executed recovery trials |
| Safety is improved | collisions, recovery failures, adequate power, deployment limits | no powered evidence or formal safety analysis | UNSUPPORTED | efficacy experiment plus explicit non-certification |
| Physical robot validation | named hardware, configuration, logs and rosbags | hardware launch and safety checklist implemented; no execution | UNSUPPORTED | staged named-TurtleBot3 run with bags/logs |
