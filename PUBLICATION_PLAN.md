# Publication Plan

## Paper-shaped scope

Target a workshop paper only after completing estimator validation and the powered V2 simulation study. The narrative is: brittle commitments under dynamic route invalidation; an operational recovery-feasibility quantity; J0–J3 causal ablation in one Nav2 stack; conditions where the treatment helps, is neutral, and harms.

## Milestones

1. **Measurement freeze:** rename the current term “local escape-option heuristic,” implement direct replan logging, executable recovery assessment, immutable manifests, and tests for every metric.
2. **Estimator study (Level 1):** generate held-out recovery rollouts; report discrimination, calibration only if probabilistic, resolution/footprint sensitivity, and pathological cases.
3. **ROS integration rerun (Level 2):** retain build/test/lifecycle/plugin-server logs at a non-null Git SHA.
4. **Static closed-stack check (Level 3 boundary):** six planners, frozen queries, path/latency only.
5. **Dynamic commissioning (Level 4):** all six planners, event observation, direct replans, bags, and executed recovery labels; tune only on development seeds.
6. **Powered evaluation (Level 4):** preregister, run paired final seeds, preserve all raw artifacts, and publish analysis scripts/figure manifests.
7. **Hardware readiness:** prepare conservative TurtleBot3 parameters, e-stop operator, speed limits, geofence, localization checks, and bag topics. Call this Level 5 only after named hardware runs exist.

## Physical TurtleBot3 checklist

Use a static known map and AMCL first; verify `/tf`, `/tf_static`, `/odom`, `/scan`, `/cmd_vel`, costmaps, footprint, timestamps, and emergency-stop path. Cap translational/angular speed and acceleration; establish a physical test perimeter and human e-stop operator; test stop command and Nav2 cancel before motion; disable automated obstacle injection; begin with J0 then zero/nonzero-weight equivalence; record parameters, robot serial/model, sensor firmware, battery, map, calibration, rosbag, ROS logs, and incident sheet. Ground truth, if externally measured, is evaluation-only.

## Release artifacts

Archive container digest/lockfile, source SHA, protocol, raw CSV/JSON, rosbag metadata and bags where feasible, analysis environment, scripts, generated figures with sidecars, exclusion log, and negative cases. Quantitative manuscript sentences must cite an artifact ID and table/figure generator.

## Submission gate

Do not submit an efficacy paper if the primary comparison is underpowered, the estimator does not predict recovery feasibility on held-out scenarios, invalid trials exceed the frozen tolerance, or figure regeneration fails. A negative-method paper remains viable if these quality gates pass.

## Three likely rejection reasons and minimum remedies

| Rejection reason | Minimum work to eliminate it |
|---|---|
| The proposed “recoverability” signal is a local heuristic without construct validity | define recovery feasibility; validate a robot-information-only estimator against held-out executed recoveries; keep local degree as an ablation |
| No powered causal evidence for H1–H4 | run all J0–J3 plus references over preregistered, balanced, multi-seed dynamic families with primary-event incidence, CIs, paired effects, and correction |
| Benchmark bias and incomplete provenance | orthogonalize event/risk/geometry factors, freeze tuning/evaluation split, retain SHA/commands/logs/bags/raw paths/costmaps and figure manifests |
