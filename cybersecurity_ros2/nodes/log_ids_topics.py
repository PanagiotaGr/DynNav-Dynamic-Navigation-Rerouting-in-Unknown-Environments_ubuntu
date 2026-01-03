#!/usr/bin/env python3
import csv
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

class IDSLogger(Node):
    """
    Logs IDS topics to CSV robustly:
    - Always writes a row at a fixed rate (even if no messages arrive)
    - Uses NaN when a topic hasn't published yet
    - Works with /ids/tf_score, /ids/tf_cusum, /ids/tf_alarm by default
    """

    def __init__(self):
        super().__init__("ids_logger")

        self.declare_parameter("score_topic", "/ids/tf_score")
        self.declare_parameter("cusum_topic", "/ids/tf_cusum")
        self.declare_parameter("alarm_topic", "/ids/tf_alarm")
        self.declare_parameter("out_csv", "tf_ids_stream.csv")
        self.declare_parameter("rate_hz", 20.0)

        self.score_topic = self.get_parameter("score_topic").value
        self.cusum_topic = self.get_parameter("cusum_topic").value
        self.alarm_topic = self.get_parameter("alarm_topic").value
        self.out_csv = self.get_parameter("out_csv").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.score = math.nan
        self.cusum = math.nan
        self.alarm = math.nan

        self.create_subscription(Float32, self.score_topic, self.cb_score, 10)
        self.create_subscription(Float32, self.cusum_topic, self.cb_cusum, 10)
        self.create_subscription(Bool, self.alarm_topic, self.cb_alarm, 10)

        self.t0 = self.get_clock().now()

        self.f = open(self.out_csv, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t_sec", "tf_score", "tf_cusum", "tf_alarm"])

        period = 1.0 / max(self.rate_hz, 1e-6)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"Logging:\n"
            f"  score: {self.score_topic}\n"
            f"  cusum: {self.cusum_topic}\n"
            f"  alarm: {self.alarm_topic}\n"
            f"to CSV: {self.out_csv} @ {self.rate_hz:.1f} Hz"
        )

    def cb_score(self, msg: Float32):
        self.score = float(msg.data)

    def cb_cusum(self, msg: Float32):
        self.cusum = float(msg.data)

    def cb_alarm(self, msg: Bool):
        self.alarm = 1.0 if bool(msg.data) else 0.0

    def tick(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        self.w.writerow([t, self.score, self.cusum, self.alarm])
        self.f.flush()

def main():
    rclpy.init()
    node = IDSLogger()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.f.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
