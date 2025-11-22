import math
import numpy as np
import csv
from shapely.geometry import Point, Polygon


def estimate_flight_time(waypoints, speed_m_s=5.0):
    total_dist = 0.0
    for i in range(1, len(waypoints)):
        x1, y1 = waypoints[i - 1]
        x2, y2 = waypoints[i]
        total_dist += math.dist((x1, y1), (x2, y2))
    return total_dist / speed_m_s


def generate_grid_waypoints(area_width_m, area_height_m, step_x, step_y):
    waypoints = []
    y_values = np.arange(0.0, area_height_m + step_y, step_y)
    x_start = 0.0
    x_end = area_width_m

    for i, y in enumerate(y_values):
        if i % 2 == 1:
            xs = np.arange(x_end, x_start - step_x, -step_x)
        else:
            xs = np.arange(x_start, x_end + step_x, step_x)
        for x in xs:
            waypoints.append((float(x), float(y)))

    return waypoints


def main():
    # Βασικά βήματα (πχ από το original planner σου)
    step_x = 22.5
    step_y = 10.0
    area_width_m = 120.0
    area_height_m = 80.0

    base_waypoints = generate_grid_waypoints(
        area_width_m, area_height_m, step_x, step_y
    )

    print(f"Base waypoints: {len(base_waypoints)}")

    # Περιοχή ενδιαφέροντος (πολύγωνο)
    aoi = Polygon([(0, 0), (120, 0), (120, 80), (0, 80)])

    # No-fly ζώνη (π.χ. ορθογώνιο στη μέση)
    no_fly = Polygon([(40, 30), (80, 30), (80, 50), (40, 50)])

    filtered = []
    for wp in base_waypoints:
        p = Point(wp[0], wp[1])
        if not aoi.contains(p):
            continue
        if no_fly.contains(p):
            continue
        filtered.append(wp)

    print(f"Waypoints after AOI + no-fly: {len(filtered)}")

    flight_time_s = estimate_flight_time(filtered, speed_m_s=5.0)
    print(f"Estimated flight time: {flight_time_s:.1f} s (~{flight_time_s/60:.1f} min)")

    with open("waypoints_poly.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x_m", "y_m"])
        for i, (x, y) in enumerate(filtered):
            writer.writerow([i, x, y])

    print("Saved waypoints_poly.csv")


if __name__ == "__main__":
    main()
