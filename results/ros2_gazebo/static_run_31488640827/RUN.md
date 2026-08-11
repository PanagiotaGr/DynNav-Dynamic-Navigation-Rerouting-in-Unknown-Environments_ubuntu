# Static Gazebo commissioning run 31488640827

- Workflow: [ROS 2 Gazebo benchmark](https://github.com/panagiotagrosdouli/DynNav/actions/runs/31488640827)
- Stable branch head: `253e3b37835f695b9363559fb823216ba8365ae2`
- GitHub PR merge revision recorded by the runner: `97ba4191c0bdfca1e3c50d00e6f84b7ff2d229b3`
- Actions artifact ID: `9100241047`
- Actions ZIP digest: `sha256:c0827b418562f57f7ceae9ec42a03c792780edae5b95785bf16c4d7847947847`
- Environment: ROS 2 Jazzy, Gazebo Sim `8.11.0`, official minimal TurtleBot3 sandbox
- Design: 6 planners × 2 scenarios × 3 measured repetitions, with one excluded warm-up query per planner
- Result: 36/36 requests succeeded; each planner had 6/6 successes

The raw `results.json` and `trials.csv` are authoritative. Pooled planning
latency means ranged from 26.33 ms (`Smac2D`) to 69.56 ms (`DynNavJoint`) in
this runner. These small commissioning samples support integration and
profiling statements only; do not infer statistical superiority from them.

All files other than this note and `SHA256SUMS` are copied byte-for-byte from
the Actions artifact.
