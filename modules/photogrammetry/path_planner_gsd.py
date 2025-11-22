import math
import numpy as np
import csv


def height_from_gsd(
    target_gsd_m: float,
    sensor_width_mm: float,
    image_width_px: int,
    focal_length_mm: float,
):
    """
    Υπολογισμός ύψους πτήσης από target GSD.
    GSD ≈ (sensor_width / image_width) * (H / f)
    => H ≈ GSD * f * image_width / sensor_width
    """
    sensor_width_m = sensor_width_mm / 1000.0
    focal_length_m = focal_length_mm / 1000.0

    H = target_gsd_m * focal_length_m * image_width_px / sensor_width_m
    return H


def camera_footprint(height_m, sensor_width_mm, sensor_height_mm, focal_length_mm):
    sensor_width_m = sensor_width_mm / 1000.0
    sensor_height_m = sensor_height_mm / 1000.0
    focal_length_m = focal_length_mm / 1000.0

    fov_x = 2 * math.atan(sensor_width_m / (2 * focal_length_m))
    fov_y = 2 * math.atan(sensor_height_m / (2 * focal_length_m))

    footprint_x = 2 * height_m * math.tan(fov_x / 2)
    footprint_y = 2 * height_m * math.tan(fov_y / 2)

    return footprint_x, footprint_y


def compute_step_sizes(footprint_x, footprint_y, frontlap=0.8, sidelap=0.7):
    step_y = footprint_y * (1.0 - frontlap)
    step_x = footprint_x * (1.0 - sidelap)
    return step_x, step_y


def generate_grid_waypoints(
    area_width_m,
    area_height_m,
    step_x,
    step_y,
    origin_x=0.0,
    origin_y=0.0,
    boustrophedon=True,
):
    waypoints = []

    y_values = np.arange(origin_y, origin_y + area_height_m + step_y, step_y)
    x_start = origin_x
    x_end = origin_x + area_width_m

    for i, y in enumerate(y_values):
        if boustrophedon and (i % 2 == 1):
            xs = np.arange(x_end, x_start - step_x, -step_x)
        else:
            xs = np.arange(x_start, x_end + step_x, step_x)

        for x in xs:
            waypoints.append((float(x), float(y)))

    return waypoints


def main():
    # ===== Απαιτήσεις εικόνας =====
    target_gsd_m = 0.02  # 2 cm/pixel

    # Αισθητήρας / κάμερα
    sensor_width_mm = 13.2
    sensor_height_mm = 8.8
    focal_length_mm = 8.8
    image_width_px = 4000
    image_height_px = 3000

    # Υπολογισμός ύψους από GSD
    height_m = height_from_gsd(
        target_gsd_m,
        sensor_width_mm,
        image_width_px,
        focal_length_mm,
    )

    print(f"Target GSD: {target_gsd_m*100:.1f} cm/pixel")
    print(f"Computed flight height: {height_m:.2f} m")

    # Overlap
    frontlap = 0.8
    sidelap = 0.7

    # Περιοχή
    area_width_m = 120.0
    area_height_m = 80.0

    footprint_x, footprint_y = camera_footprint(
        height_m, sensor_width_mm, sensor_height_mm, focal_length_mm
    )
    print(f"Camera footprint: {footprint_x:.2f} m x {footprint_y:.2f} m")

    step_x, step_y = compute_step_sizes(
        footprint_x, footprint_y, frontlap=frontlap, sidelap=sidelap
    )
    print(f"Step X: {step_x:.2f} m, Step Y: {step_y:.2f} m")

    waypoints = generate_grid_waypoints(
        area_width_m, area_height_m, step_x, step_y, 0.0, 0.0, True
    )

    print(f"Total waypoints: {len(waypoints)}")

    with open("waypoints_gsd.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x_m", "y_m"])
        for i, (x, y) in enumerate(waypoints):
            writer.writerow([i, x, y])

    print("Saved waypoints_gsd.csv")


if __name__ == "__main__":
    main()
