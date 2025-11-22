#!/usr/bin/env python3
import math
import time
from typing import List
import os
import sys

# === Add project root to Python path ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose

from photogrammetry_module.optimal_path_planner import PhotogrammetryPathPlanner
from photogrammetry_module.image_capture import ImageCapture


def yaw_to_quaternion(yaw: float):
    """
    Convert yaw [rad] to quaternion (x, y, z, w) for planar motion.
    """
    half_yaw = yaw / 2.0
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


class PhotogrammetryMissionNode(Node):
    def __init__(self):
        super().__init__("photogrammetry_mission_node")

        # --- PARAMETERS ---
        # μικρή ακτίνα γύρω από το robot
        self.declare_parameter("radius", 0.6)
        self.declare_parameter("height", 0.3)
        self.declare_parameter("num_views", 10)

        radius = float(self.get_parameter("radius").value)
        height = float(self.get_parameter("height").value)
        num_views = int(self.get_parameter("num_views").value)

        self.get_logger().info(
            f"Starting photogrammetry mission with "
            f"radius={radius:.2f}, height={height:.2f}, num_views={num_views}"
        )

        # Path planner
        self.planner = PhotogrammetryPathPlanner(
            radius=radius,
            height=height,
            num_views=num_views
        )

        # Image capture
        self.image_capture = ImageCapture(device_id=0)

        # Nav2 action client
        self._action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Θα γεμίσουν όταν έχουμε amcl_pose
        self.center_x = None
        self.center_y = None
        self.waypoints: List[PoseStamped] = []
        self.current_index = 0

        self.have_initial_pose = False
        self.server_ready = False
        self.mission_started = False

        # Συνδρομή στο amcl_pose για να πάρουμε τη θέση του robot στο map frame
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "amcl_pose",
            self.amcl_callback,
            10
        )

        # Περιμένουμε Nav2 action server σε ξεχωριστό thread
        self._wait_for_server_async()

    # ----------------- AMCL / POSE -----------------

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        if not self.have_initial_pose:
            self.center_x = float(msg.pose.pose.position.x)
            self.center_y = float(msg.pose.pose.position.y)
            self.have_initial_pose = True

            self.get_logger().info(
                f"Received initial pose from AMCL, "
                f"center=({self.center_x:.2f}, {self.center_y:.2f})"
            )

            # Αν είναι έτοιμος και ο Nav2 server, μπορούμε να ξεκινήσουμε
            self._maybe_start_mission()

    # ----------------- NAV2 SERVER WAIT -----------------

    def _wait_for_server_async(self):
        self.get_logger().info("Waiting for Nav2 'navigate_to_pose' action server...")

        def _wait():
            # loop με μικρό timeout μέχρι να είναι διαθέσιμος
            while not self._action_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().warn(
                    "Nav2 'navigate_to_pose' action server not available yet, retrying..."
                )
                time.sleep(1.0)

            self.server_ready = True
            self.get_logger().info("Nav2 action server is up.")
            self._maybe_start_mission()

        # πολύ απλό "fake" async: χρησιμοποιούμε timer για να μην μπλοκάρουμε spin
        self.create_timer(0.1, lambda: None)  # κρατά το node ζωντανό
        # ξεκινάμε blocking wait σε ξεχωριστό thread
        import threading
        threading.Thread(target=_wait, daemon=True).start()

    # ----------------- MISSION START LOGIC -----------------

    def _maybe_start_mission(self):
        """
        Ξεκινάει την αποστολή μόνο όταν:
        - έχουμε πάρει αρχική pose από amcl (center_x, center_y)
        - έχει σηκωθεί ο Nav2 action server
        - δεν έχουμε ήδη ξεκινήσει τη mission
        """
        if self.mission_started:
            return
        if not self.server_ready:
            return
        if not self.have_initial_pose:
            self.get_logger().info("Waiting for initial pose from AMCL...")
            return

        self.get_logger().info(
            f"Starting photogrammetry mission around "
            f"({self.center_x:.2f}, {self.center_y:.2f})"
        )

        self.waypoints = self._generate_waypoints(self.center_x, self.center_y)
        self.current_index = 0
        self.mission_started = True

        if not self.waypoints:
            self.get_logger().error("No waypoints generated, aborting mission.")
            return

        self._send_next_goal()

    # ----------------- WAYPOINT GENERATION -----------------

    def _generate_waypoints(self, cx: float, cy: float) -> List[PoseStamped]:
        raw_points = self.planner.generate_circular_path(cx, cy)
        waypoints: List[PoseStamped] = []

        for (x, y, z, yaw) in raw_points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()

            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)

            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            pose.pose.orientation.w = float(qw)

            waypoints.append(pose)

        self.get_logger().info(f"Generated {len(waypoints)} waypoints for photogrammetry")
        if waypoints:
            p0 = waypoints[0].pose.position
            self.get_logger().info(
                f"First waypoint at ({p0.x:.2f}, {p0.y:.2f}) in map frame"
            )
        return waypoints

    # ----------------- NAV2 GOAL SENDING -----------------

    def _send_next_goal(self):
        if self.current_index >= len(self.waypoints):
            self.get_logger().info("Photogrammetry mission complete. All waypoints visited.")
            return

        goal_pose = self.waypoints[self.current_index]
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            f"Sending goal {self.current_index + 1}/{len(self.waypoints)}: "
            f"x={goal_pose.pose.position.x:.2f}, "
            f"y={goal_pose.pose.position.y:.2f}"
        )

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # μπορείς να log-άρεις remaining distance κτλ αν θες
        # self.get_logger().debug(f"Remaining distance: {feedback.distance_remaining:.2f} m")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(
                f"Goal {self.current_index + 1} was rejected by Nav2."
            )
            self.current_index += 1
            time.sleep(0.5)
            self._send_next_goal()
            return

        self.get_logger().info("Goal accepted, waiting for result...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result()
        status = result.status

        if status == 4:  # SUCCEEDED
            self.get_logger().info(
                f"Reached waypoint {self.current_index + 1}. Capturing image..."
            )
            try:
                image_path = self.image_capture.capture()
                self.get_logger().info(f"Captured image at {image_path}")
            except Exception as e:
                self.get_logger().error(f"Image capture failed: {e}")
        else:
            self.get_logger().warn(
                f"Goal at waypoint {self.current_index + 1} failed with status {status}"
            )

        self.current_index += 1
        time.sleep(1.0)
        self._send_next_goal()


def main(args=None):
    rclpy.init(args=args)
    node = PhotogrammetryMissionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
