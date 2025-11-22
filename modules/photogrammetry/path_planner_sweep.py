import math
import numpy as np
import csv


def camera_footprint(
    height_m,
    sensor_width_mm,
    sensor_height_mm,
    focal_length_mm,
):
    sensor_width_m = sensor_width_mm / 1000.0
    sensor_height_m = sensor_height_mm / 1000.0
    focal_length_m = focal_length_mm / 1000.0

    fov_x = 2 * math.atan(sensor_width_m / (2 * focal_length_m))
    fov_y = 2 * math.atan(sensor_height_m / (2 * focal_length_m))

    footprint_x = 2 * height_m * math.tan(fov_x / 2)
    footprint_y = 2 * height_m * math.tan(fov_y / 2)

    return footprint_x, footprint_y


def compute_step_sizes(footprint_x, footprint_y, frontlap, sidelap):
    step_y = footprint_y * (1.0 - frontlap)
    step_x = footprint_x * (1.0 - sidelap)
    return step_x, step_y


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


def estimate_flight_time(waypoints, speed_m_s=5.0):
    total_dist = 0.0
    for i in range(1, len(waypoints)):
        x1, y1 = waypoints[i - 1]
        x2, y2 = waypoints[i]
        total_dist += math.dist((x1, y1), (x2, y2))
    return total_dist / speed_m_s


def main():
    # Σταθερές παράμετροι κάμερας / πτήσης
    height_m = 50.0
    sensor_width_mm = 13.2
    sensor_height_mm = 8.8
    focal_length_mm = 8.8
    image_width_px = 4000
    image_height_px = 3000

    area_width_m = 120.0
    area_height_m = 80.0

    footprint_x, footprint_y = camera_footprint(
        height_m,
        sensor_width_mm,
        sensor_height_mm,
        focal_length_mm,
    )

    print(f"Camera footprint: {footprint_x:.2f} m x {footprint_y:.2f} m")

    # Τιμές για sweep
    frontlaps = [0.6, 0.7, 0.8, 0.9]
    sidelaps = [0.6, 0.7, 0.8, 0.9]

    rows = []

    for fl in frontlaps:
        for sl in sidelaps:
            step_x, step_y = compute_step_sizes(
                footprint_x, footprint_y, frontlap=fl, sidelap=sl
            )

            waypoints = generate_grid_waypoints(
                area_width_m, area_height_m, step_x, step_y
            )

            flight_time_s = estimate_flight_time(waypoints, speed_m_s=5.0)

            print(
                f"frontlap={fl:.1f}, sidelap={sl:.1f} -> "
                f"{len(waypoints)} WPs, time={flight_time_s:.1f}s"
            )

            rows.append(
                {
                    "frontlap": fl,
                    "sidelap": sl,
                    "step_x_m": step_x,
                    "step_y_m": step_y,
                    "num_waypoints": len(waypoints),
                    "flight_time_s": flight_time_s,
                    "flight_time_min": flight_time_s / 60.0,
                }
            )

    # Αποθήκευση σε CSV
    with open("planner_sweep_results.csv", "w", newline="") as f:
        fieldnames = [
            "frontlap",
            "sidelap",
            "step_x_m",
            "step_y_m",
            "num_waypoints",
            "flight_time_s",
            "flight_time_min",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Saved planner_sweep_results.csv")


if __name__ == "__main__":
    main()

