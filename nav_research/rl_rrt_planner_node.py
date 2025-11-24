import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from .advanced_planners import (
    train_q_learning,
    extract_policy_path,
)


class RLPlannerNode(Node):
    """
    RL-based Path Planner (GridWorld + Q-learning) με ROS2 interface.

    - Εκπαιδεύει μία φορά Q-learning agent σε GridWorld.
    - Περιμένει goal από /goal_pose (PoseStamped).
    - Υπολογίζει διαδρομή στο grid με το RL policy.
    - Δημοσιεύει nav_msgs/Path στο /planned_path.

    Σημείωση:
    - Το περιβάλλον είναι 2D grid [0..grid_size-1] x [0..grid_size-1].
    - Θεωρούμε fixed start στο (0, 0).
    - Το goal από RViz προβάλλεται σε κοντινό grid cell.
    """

    def __init__(self):
        super().__init__("rl_rrt_planner")

        # Παράμετροι
        self.declare_parameter("grid_size", 10)
        self.grid_size = self.get_parameter("grid_size").get_parameter_value().integer_value
        if self.grid_size <= 1:
            self.grid_size = 10

        self.get_logger().info(f"[RLPlannerNode] Initializing with grid_size={self.grid_size}")

        # Εκπαίδευση Q-learning agent σε GridWorld
        # (χρησιμοποιεί internally GridWorldEnv από advanced_planners)
        episodes = 500
        self.env, self.agent = train_q_learning(
            episodes=episodes,
            width=self.grid_size,
            height=self.grid_size,
            start=(0, 0),
            goal=(self.grid_size - 1, self.grid_size - 1),
            obstacles=None,  # μπορείς να βάλεις εμπόδια αν θέλεις
        )

        self.get_logger().info(f"[RLPlannerNode] Q-learning training done after {episodes} episodes.")

        # Publisher για path
        self.path_pub = self.create_publisher(Path, "/planned_path", 10)

        # Subscriber για goal από RViz / άλλο node
        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_callback,
            10,
        )

        self.get_logger().info("[RLPlannerNode] Waiting for /goal_pose messages...")


    def goal_callback(self, msg: PoseStamped):
        """
        Callback όταν έρχεται νέο goal από /goal_pose.
        """
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        self.get_logger().info(
            f"[RLPlannerNode] Received goal pose: x={gx:.2f}, y={gy:.2f}"
        )

        # Προβολή goal σε grid [0 .. grid_size-1]
        goal_cell = self._world_to_grid(gx, gy)
        self.get_logger().info(f"[RLPlannerNode] Mapped goal to grid cell {goal_cell}")

        # (Προσεγγιστικά) ενημερώνουμε το goal του περιβάλλοντος
        self.env.goal = goal_cell

        # Επαναφορά στο start
        self.env.reset()

        # Εξαγωγή path με το policy του agent
        grid_path = extract_policy_path(self.env, self.agent, max_steps=self.grid_size * self.grid_size)

        self.get_logger().info(f"[RLPlannerNode] RL grid path length = {len(grid_path)}")

        # Μετατροπή από grid path -> nav_msgs/Path
        path_msg = self._grid_path_to_ros_path(grid_path)

        self.path_pub.publish(path_msg)
        self.get_logger().info("[RLPlannerNode] Published RL-based /planned_path")


    def _world_to_grid(self, x: float, y: float):
        """
        Πολύ απλή χαρτογράφηση world → grid:
        Θεωρούμε ότι ο χρήστης κλικάρει κοντά στο [0, grid_size-1].

        - Κόβουμε (clamp) στις άκρες.
        - Στρογγυλοποιούμε στο κοντινότερο cell.
        """
        gx = int(round(x))
        gy = int(round(y))

        gx = max(0, min(self.grid_size - 1, gx))
        gy = max(0, min(self.grid_size - 1, gy))

        return (gx, gy)


    def _grid_path_to_ros_path(self, grid_path):
        """
        Μετατρέπει μια λίστα από grid states [(ix, iy), ...]
        σε nav_msgs/Path στο frame "map".
        Υποθέτουμε cell_size = 1.0 (1 unit = 1 meter).
        """
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for (ix, iy) in grid_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(ix)
            pose.pose.position.y = float(iy)
            # Τα orientation τα αφήνουμε default (0,0,0,1)
            path_msg.poses.append(pose)

        return path_msg


def main(args=None):
    rclpy.init(args=args)
    node = RLPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

