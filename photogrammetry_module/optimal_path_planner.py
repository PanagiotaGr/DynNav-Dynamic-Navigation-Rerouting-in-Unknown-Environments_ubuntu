import numpy as np
from math import sin, cos, pi


class PhotogrammetryPathPlanner:
    """
    Simple visibility-based path planner for photogrammetry missions.

    Generates a set of viewpoints (x, y, z, yaw) arranged around a target
    region or object. The idea is to ensure:
    - sufficient coverage of the object
    - high overlap between consecutive images
    - inward-looking camera orientation.
    """

    def __init__(self,
                 radius: float = 1.5,
                 height: float = 0.3,
                 num_views: int = 24):
        """
        :param radius: distance of the robot/camera from the target center (m)
        :param height: camera height in the world frame (m)
        :param num_views: number of viewpoints on the circle
        """
        self.radius = radius
        self.height = height
        self.num_views = num_views

    def generate_circular_path(self, center_x: float = 0.0, center_y: float = 0.0):
        """
        Generate a circular path of viewpoints around (center_x, center_y).

        :return: np.ndarray of shape (N, 4) with [x, y, z, yaw]
        """
        waypoints = []

        for i in range(self.num_views):
            theta = 2.0 * pi * (i / self.num_views)

            x = center_x + self.radius * cos(theta)
            y = center_y + self.radius * sin(theta)
            z = self.height

            # yaw so that the robot looks toward the center
            yaw = theta + pi

            waypoints.append([x, y, z, yaw])

        return np.array(waypoints)

    def export_waypoints(self,
                         filename: str = "photogrammetry_waypoints.csv",
                         center_x: float = 0.0,
                         center_y: float = 0.0):
        """
        Save waypoints to csv for offline analysis or benchmarking.
        """
        path = self.generate_circular_path(center_x, center_y)
        np.savetxt(filename, path, delimiter=",")
        print(f"[PhotogrammetryPathPlanner] Exported path to {filename}")
