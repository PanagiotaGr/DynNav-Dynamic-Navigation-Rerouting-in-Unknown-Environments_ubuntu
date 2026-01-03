#!/usr/bin/env python3
import csv
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import OccupancyGrid

class CostmapIntegrityMonitor(Node):
    def __init__(self):
        super().__init__("costmap_integrity_monitor")

        self.declare_parameter("topic", "/map_spoofed")
        self.declare_parameter("occ_thresh", 65)         # occupancy considered obstacle
        self.declare_parameter("flip_rate_thresh", 0.02) # fraction of cells flipping per frame
        self.declare_parameter("density_delta_thresh", 0.02)
        self.declare_parameter("out_csv", "costmap_ids_log.csv")

        self.topic = self.get_parameter("topic").value
        self.occ_thresh = int(self.get_parameter("occ_thresh").value)
        self.flip_thr = float(self.get_parameter("flip_rate_thresh").value)
        self.dens_thr = float(self.get_parameter("density_delta_thresh").value)

        self.pub_alarm = self.create_publisher(Bool, "/ids/costmap_alarm", 10)
        self.pub_score = self.create_publisher(Float32, "/ids/costmap_score", 10)

        self.prev = None
        self.t0 = self.get_clock().now()

        self.out_csv = self.get_parameter("out_csv").value
        self.f = open(self.out_csv, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t_sec","obs_density","flip_rate","score","alarm"])

        self.sub = self.create_subscription(OccupancyGrid, self.topic, self.cb, 10)
        self.get_logger().info(f"Monitoring costmap: {self.topic}, logging to {self.out_csv}")

    def cb(self, msg: OccupancyGrid):
        data = np.array(msg.data, dtype=np.int16)
        # unknown (-1) ignored for density
        known = data[data >= 0]
        if known.size == 0:
            return

        obs = (known >= self.occ_thresh).astype(np.float32)
        density = float(obs.mean())

        flip_rate = 0.0
        if self.prev is not None and self.prev.shape == data.shape:
            prev_known = self.prev[self.prev >= 0]
            now_known = data[data >= 0]
            n = min(prev_known.size, now_known.size)
            if n > 0:
                prev_obs = (prev_known[:n] >= self.occ_thresh)
                now_obs  = (now_known[:n]  >= self.occ_thresh)
                flip_rate = float(np.mean(prev_obs != now_obs))

        # score: combine density jump + flip rate
        dens_delta = 0.0 if self.prev is None else abs(density - float(((self.prev[self.prev>=0] >= self.occ_thresh).mean())))
        score = max(flip_rate / max(self.flip_thr,1e-9), dens_delta / max(self.dens_thr,1e-9))
        alarm = (flip_rate > self.flip_thr) or (dens_delta > self.dens_thr)

        self.prev = data.copy()

        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        self.w.writerow([t, density, flip_rate, score, 1 if alarm else 0])
        self.f.flush()

        self.pub_score.publish(Float32(data=float(score)))
        self.pub_alarm.publish(Bool(data=bool(alarm)))

def main():
    rclpy.init()
    node = CostmapIntegrityMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
