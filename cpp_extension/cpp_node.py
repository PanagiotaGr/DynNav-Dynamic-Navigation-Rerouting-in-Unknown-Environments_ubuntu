#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from cpp_manager import CPPManager


class CPPNode(Node):
    def __init__(self):
        super().__init__('cpp_node')

        # Subscriptions
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Υποθέτω ότι έχεις ένα topic PoseStamped με pose ρομπότ (π.χ. /robot_pose)
        # Αν όχι και έχεις μόνο /odom ή tf, το προσαρμόζεις μετά.
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/robot_pose',
            self.pose_callback,
            10
        )

        # Publisher για goal που θα δίνεις στο navigation σου
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/cpp_goal',
            10
        )

        # Εσωτερικές μεταβλητές
        self.occupancy = None
        self.map_info = None
        self.cpp_manager = None
        self.current_pose = None

        # Timer: κάθε 2s αποφασίζουμε επόμενο goal
        self.timer = self.create_timer(2.0, self.timer_callback)

        self.get_logger().info('CPPNode started')

    def map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        # Μετατροπή occupancy σε numpy
        data = np.array(msg.data, dtype=np.int16).reshape((height, width))

        self.occupancy = data
        self.map_info = msg.info

        # Αν δεν έχουμε ήδη cpp_manager, τον φτιάχνουμε τώρα
        if self.cpp_manager is None:
            self.cpp_manager = CPPManager(width, height, resolution)
            self.get_logger().info(
                f'CPPManager initialized: {width}x{height}, res={resolution}'
            )

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg

        if self.cpp_manager is None or self.map_info is None:
            return

        # Παίρνουμε x, y σε world coords
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Update coverage
        self.cpp_manager.update_pose(x, y)

    def timer_callback(self):
        if self.cpp_manager is None or self.occupancy is None or self.current_pose is None:
            return

        # Ρωτάμε τον CPP manager για επόμενο goal
        goal_cell = self.cpp_manager.decide_next_goal(self.occupancy)

        if goal_cell is None:
            self.get_logger().info('No more frontiers – coverage likely complete.')
            return

        gx_cell, gy_cell = goal_cell

        # Μετατροπή από cell -> world coordinates
        res = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        gx = gx_cell * res + origin_x
        gy = gy_cell * res + origin_y

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.map_info.map_load_time.clock_type.name if hasattr(self.map_info, 'map_load_time') else 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = gx
        goal_msg.pose.position.y = gy
        goal_msg.pose.position.z = 0.0

        # Orientation: για αρχή, μηδενική yaw (ή κράτα την τωρινή orientation)
        goal_msg.pose.orientation = self.current_pose.pose.orientation

        self.goal_pub.publish(goal_msg)
        self.get_logger().info(
            f'Published CPP goal at world coords: ({gx:.2f}, {gy:.2f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CPPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

