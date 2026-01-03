#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class MinimalTFPublisher(Node):
    """
    Publishes a clean, kinematically-consistent TF:
    world -> base_link
    Used as ground-truth motion for IDS experiments.
    """

    def __init__(self):
        super().__init__("minimal_tf_publisher")

        self.br = TransformBroadcaster(self)

        self.declare_parameter("parent_frame", "world")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("v", 0.1)     # m/s
        self.declare_parameter("w", 0.05)    # rad/s

        self.parent = self.get_parameter("parent_frame").value
        self.child = self.get_parameter("child_frame").value
        self.v = float(self.get_parameter("v").value)
        self.w = float(self.get_parameter("w").value)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_t = self.get_clock().now()

        self.timer = self.create_timer(0.05, self.step)  # 20 Hz
        self.get_logger().info(f"Publishing TF {self.parent} -> {self.child}")

    def step(self):
        now = self.get_clock().now()
        dt = (now - self.last_t).nanoseconds * 1e-9
        self.last_t = now

        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.parent
        t.child_frame_id = self.child

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        qz = math.sin(self.yaw * 0.5)
        qw = math.cos(self.yaw * 0.5)
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.br.sendTransform(t)

def main():
    rclpy.init()
    node = MinimalTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
