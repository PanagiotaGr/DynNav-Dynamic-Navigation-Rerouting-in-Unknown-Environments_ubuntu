#!/usr/bin/env python3
import csv
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist

class PlannerHealthMonitor(Node):
    def __init__(self):
        super().__init__("planner_health_monitor")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("wz_spike", 1.5)       # rad/s
        self.declare_parameter("vx_spike", 0.6)       # m/s
        self.declare_parameter("osc_window_sec", 4.0) # lookback
        self.declare_parameter("osc_flip_thresh", 6)  # sign flips in wz
        self.declare_parameter("out_csv", "planner_health_log.csv")

        self.cmd_topic = self.get_parameter("cmd_vel_topic").value
        self.wz_spike = float(self.get_parameter("wz_spike").value)
        self.vx_spike = float(self.get_parameter("vx_spike").value)
        self.win = float(self.get_parameter("osc_window_sec").value)
        self.flip_thr = int(self.get_parameter("osc_flip_thresh").value)

        self.pub_alarm = self.create_publisher(Bool, "/ids/planner_alarm", 10)
        self.pub_score = self.create_publisher(Float32, "/ids/planner_score", 10)

        self.t0 = self.get_clock().now()
        self.hist_t = []
        self.hist_wz = []

        self.out_csv = self.get_parameter("out_csv").value
        self.f = open(self.out_csv, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t_sec","vx","wz","wz_flips","score","alarm"])

        self.sub = self.create_subscription(Twist, self.cmd_topic, self.cb, 50)
        self.timer = self.create_timer(0.1, self.tick)

        self.latest_vx = 0.0
        self.latest_wz = 0.0

        self.get_logger().info(f"Monitoring {self.cmd_topic}, logging to {self.out_csv}")

    def cb(self, msg: Twist):
        self.latest_vx = float(msg.linear.x)
        self.latest_wz = float(msg.angular.z)
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        self.hist_t.append(t)
        self.hist_wz.append(self.latest_wz)

    def count_sign_flips(self, wz):
        s = np.sign(wz)
        flips = np.sum((s[1:] * s[:-1]) < 0)
        return int(flips)

    def tick(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9

        # keep only last window
        while self.hist_t and (t - self.hist_t[0] > self.win):
            self.hist_t.pop(0)
            self.hist_wz.pop(0)

        wz_flips = 0
        if len(self.hist_wz) >= 5:
            wz_flips = self.count_sign_flips(np.array(self.hist_wz))

        spike = (abs(self.latest_wz) > self.wz_spike) or (abs(self.latest_vx) > self.vx_spike)
        osc = (wz_flips >= self.flip_thr)

        # score: max of spike ratios + oscillation ratio
        score = max(abs(self.latest_wz)/max(self.wz_spike,1e-9),
                    abs(self.latest_vx)/max(self.vx_spike,1e-9),
                    wz_flips/max(self.flip_thr,1e-9))

        alarm = spike or osc

        self.w.writerow([t, self.latest_vx, self.latest_wz, wz_flips, score, 1 if alarm else 0])
        self.f.flush()

        self.pub_score.publish(Float32(data=float(score)))
        self.pub_alarm.publish(Bool(data=bool(alarm)))

def main():
    rclpy.init()
    node = PlannerHealthMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
