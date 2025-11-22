#!/usr/bin/env python3
import math
import time
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
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

        # Parameters – μπορείς να τα κάνεις ROS2 parameters αν θέλεις
        self.declare_parameter("radius", 1.5)
        self.declare_parameter("height", 0.3)
        self.declare_parameter("num_views", 24)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)

        radius = self.get_parameter("radius").value
        height = self.get_parameter("height").value
        num_views = self.get_parameter("num_views").value
        center_x = self.get_parameter("center_x").value
        center_y = self.get_parameter("center_y").value

        self.get_logger().info(
            f"Starting photogrammetry mission with radius={radius}, "
            f"height={height}, num_views={num_views}"
        )

        # Path planner
        self.planner = PhotogrammetryPathPlanner(
            radius=radius,
            height=height,
            num_views=num_views
        )

        # Image capture utility
        self.image_capture = ImageCapture(device_id=0)

        # Nav2 action client
        self._action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.waypoints = self._generate_waypoints(center_x, center_y)
        self.current_index = 0

        # Ξεκίνα μόλις είναι έτοιμος ο Nav2 action server
        self._wait_for_server_and_start()

    def _generate_waypoints(self, cx: float, cy: float) -> List[PoseStamped]:
        raw_points = self.planner.generate_circular_path(cx, cy)
        waypoints = []

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
        return waypoints

    def _wait_for_server_and_start(self):
        self.get_logger().info("Waiting for Nav2 'navigate_to_pose' action server...")
        if not self._action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("Nav2 action server not available after 30 seconds.")
            return
        self.get_logger().info("Nav2 action server is up. Starting mission...")
        self._send_next_goal()

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
        # Αν θες, μπορείς να log-άρεις απόσταση κ.λπ.
        # self.get_logger().debug(f"Remaining distance: {feedback.distance_remaining:.2f} m")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by Nav2.")
            self.current_index += 1
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
        time.sleep(1.0)  # μικρή παύση πριν από το επόμενο goal
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
