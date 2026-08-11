# Retained ROS 2 Jazzy / Gazebo Harmonic evidence

This directory makes the passing GitHub Actions commissioning artifacts
permanent and reviewable after the default Actions retention window expires.

| Run | Scope | Contract result | Evidence directory |
|---|---|---|---|
| [31488640827](https://github.com/panagiotagrosdouli/DynNav/actions/runs/31488640827) | Static `ComputePathToPose`, 6 planners × 2 scenarios × 3 repetitions | 36/36 successful requests | [`static_run_31488640827/`](static_run_31488640827/) |
| [31488640894](https://github.com/panagiotagrosdouli/DynNav/actions/runs/31488640894) | Dynamic `NavigateToPose`, 4 planners × 2 frozen events × 1 repetition | 8/8 valid trials; 7 successes and 1 genuine timeout | [`dynamic_run_31488640894/`](dynamic_run_31488640894/) |

The static run is planner-server integration and path/latency evidence. The
dynamic run is an `n=1` protocol commissioning result. Neither is a powered
comparative study, physical-robot evidence, a formal safety proof, or evidence
of certified collision avoidance. In the dynamic smoke run, no planner suffered
an operational-irreversibility failure, so it does not establish the central H1
reduction claim.

Every directory contains the exact Actions artifact payload plus a human-readable
run note and `SHA256SUMS`. The `source_revision` inside result JSON is GitHub's
ephemeral pull-request merge SHA; the stable branch head is recorded in each run
note.
