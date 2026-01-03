#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

class CostmapSpoofInjector(Node):
    def __init__(self):
        super().__init__("costmap_spoof_injector")

        self.declare_parameter("in_topic", "/map")
        self.declare_parameter("out_topic", "/map_spoofed")
        self.declare_parameter("attack_start_sec", 8.0)
        self.declare_parameter("mode", "blob")  # blob | stripe
        self.declare_parameter("blob_radius_cells", 6)
        self.declare_parameter("blob_value", 100)
        self.declare_parameter("center_x_frac", 0.6)  # center as fraction of width
        self.declare_parameter("center_y_frac", 0.5)

        self.in_topic = self.get_parameter("in_topic").value
        self.out_topic = self.get_parameter("out_topic").value
        self.attack_start = float(self.get_parameter("attack_start_sec").value)
        self.mode = self.get_parameter("mode").value

        self.sub = self.create_subscription(OccupancyGrid, self.in_topic, self.cb, 10)
        self.pub = self.create_publisher(OccupancyGrid, self.out_topic, 10)

        self.t0 = self.get_clock().now()
        self.latest = None
        self.get_logger().info(f"Injecting from {self.in_topic} -> {self.out_topic} mode={self.mode}")

    def cb(self, msg: OccupancyGrid):
        self.latest = msg
        self.publish()

    def publish(self):
        if self.latest is None:
            return

        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        out = OccupancyGrid()
        out.header = self.latest.header
        out.info = self.latest.info
        out.data = list(self.latest.data)

        if t < self.attack_start:
            self.pub.publish(out)
            return

        w = out.info.width
        h = out.info.height
        grid = np.array(out.data, dtype=np.int16).reshape((h, w))

        if self.mode == "blob":
            cx = int(float(self.get_parameter("center_x_frac").value) * w)
            cy = int(float(self.get_parameter("center_y_frac").value) * h)
            r = int(self.get_parameter("blob_radius_cells").value)
            val = int(self.get_parameter("blob_value").value)
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
            grid[mask] = val

        elif self.mode == "stripe":
            val = int(self.get_parameter("blob_value").value)
            grid[:, w//2:w//2+3] = val

        out.data = grid.reshape(-1).tolist()
        self.pub.publish(out)

def main():
    rclpy.init()
    node = CostmapSpoofInjector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
