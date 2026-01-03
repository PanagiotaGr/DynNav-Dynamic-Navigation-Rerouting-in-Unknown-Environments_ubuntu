#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

class SecurityFusionLayer(Node):
    def __init__(self):
        super().__init__("security_fusion_layer")

        self.declare_parameter("latch_steps", 80)
        self.declare_parameter("vote_k", 2)  # k-out-of-n voting
        self.declare_parameter("trust_decay", 0.02)
        self.declare_parameter("trust_recover", 0.005)

        self.latch_steps = int(self.get_parameter("latch_steps").value)
        self.vote_k = int(self.get_parameter("vote_k").value)
        self.decay = float(self.get_parameter("trust_decay").value)
        self.recover = float(self.get_parameter("trust_recover").value)

        self.tf_alarm = False
        self.cm_alarm = False
        self.pl_alarm = False

        self.trust = 1.0
        self.latch = 0

        self.pub_sys = self.create_publisher(Bool, "/ids/system_alarm", 10)
        self.pub_safe = self.create_publisher(Bool, "/nav/safe_mode", 10)
        self.pub_trust = self.create_publisher(Float32, "/ids/trust_signal", 10)

        self.create_subscription(Bool, "/ids/tf_alarm", self.cb_tf, 10)
        self.create_subscription(Bool, "/ids/costmap_alarm", self.cb_cm, 10)
        self.create_subscription(Bool, "/ids/planner_alarm", self.cb_pl, 10)

        self.timer = self.create_timer(0.1, self.tick)

        self.get_logger().info("Security fusion layer online (k-out-of-n voting + trust).")

    def cb_tf(self, m): self.tf_alarm = bool(m.data)
    def cb_cm(self, m): self.cm_alarm = bool(m.data)
    def cb_pl(self, m): self.pl_alarm = bool(m.data)

    def tick(self):
        votes = sum([self.tf_alarm, self.cm_alarm, self.pl_alarm])
        system_alarm = votes >= self.vote_k

        if system_alarm:
            self.latch = self.latch_steps
            self.trust = max(0.0, self.trust - self.decay)
        else:
            if self.latch > 0:
                self.latch -= 1
            self.trust = min(1.0, self.trust + self.recover)

        safe_mode = self.latch > 0

        self.pub_sys.publish(Bool(data=system_alarm))
        self.pub_safe.publish(Bool(data=safe_mode))
        self.pub_trust.publish(Float32(data=float(self.trust)))

def main():
    rclpy.init()
    node = SecurityFusionLayer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
