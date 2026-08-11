# Dynamic Gazebo commissioning run 31488640894

- Workflow: [ROS 2 dynamic route-invalidation benchmark](https://github.com/panagiotagrosdouli/DynNav/actions/runs/31488640894)
- Stable branch head: `253e3b37835f695b9363559fb823216ba8365ae2`
- GitHub PR merge revision recorded by the runner: `97ba4191c0bdfca1e3c50d00e6f84b7ff2d229b3`
- Actions artifact ID: `9100441749`
- Actions ZIP digest: `sha256:fb304c7bf49a85316aa0715aa6bf1fe95ce8df6b96109e621009e0a3158ac9b4`
- Environment: ROS 2 Jazzy, Gazebo Sim `8.11.0`, official minimal TurtleBot3 sandbox
- Design: 4 planners × 2 frozen obstacle events × 1 repetition
- Measurement contract: 8/8 valid trials, blocker observed in 8/8 post-event costmaps, minimum injection clearance 0.955 m against a frozen 0.90 m floor
- Navigation result: 7 successes; one valid `DynNavJoint` execution timeout in the forward-closure negative control
- Operational-irreversibility result: 0/8 failures under the declared eight-connected inflated-costmap recovery contract

This is a protocol commissioning run, not a powered comparative result. The
absence of operational-irreversibility failures leaves no event with which to
estimate or compare failure reduction. The genuine timeout is retained as a
negative outcome rather than removed.

All files other than this note and `SHA256SUMS` are copied byte-for-byte from
the Actions artifact.
