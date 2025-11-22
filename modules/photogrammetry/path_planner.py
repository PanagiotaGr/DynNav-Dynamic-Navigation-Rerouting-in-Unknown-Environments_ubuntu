import math
import numpy as np
import csv


def camera_footprint(
    height_m: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    focal_length_mm: float,
):
    """
    Υπολογίζει το αποτύπωμα της κάμερας στο έδαφος (σε μέτρα),
    με βάση το pinhole camera model.
    """

    sensor_width_m = sensor_width_mm / 1000.0
    sensor_height_m = sensor_height_mm / 1000.0
    focal_length_m = focal_length_mm / 1000.0

    fov_x = 2 * math.atan(sensor_width_m / (2 * focal_length_m))
    fov_y = 2 * math.atan(sensor_height_m / (2 * focal_length_m))

    footprint_x = 2 * height_m * math.tan(fov_x / 2)
    footprint_y = 2 * height_m * math.tan(fov_y / 2)

    return footprint_x, footprint_y


def compute_step_sizes(
    footprint_x: float,
    footprint_y: float,
    frontlap: float = 0.8,
    sidelap: float = 0.7,
):
    """
    Υπολογίζει την απόσταση μεταξύ διαδοχικών λήψεων
    ώστε να ικανοποιούνται τα ποσοστά frontlap/sidelap.
    """
    step_y = footprint_y * (1.0 - frontlap)   # κατά μήκος της πορείας
    step_x = footprint_x * (1.0 - sidelap)    # μεταξύ γειτονικών γραμμών

    return step_x, step_y


def generate_grid_waypoints(
    area_width_m: float,
    area_height_m: float,
    step_x: float,
    step_y: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    boustrophedon: bool = True,
):
    """
    Δημιουργεί πλέγμα waypoint για κάλυψη περιοχής.
    Αν boustrophedon=True, οι γραμμές είναι ζιγκ-ζαγκ ώστε
    να μειώνονται οι περιττές στροφές.
    """
    import numpy as np

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
    # ===== Παράμετροι κάμερας & πτήσης =====
    height_m = 50.0  # ύψος πτήσης

    # Παράδειγμα μικρού UAV αισθητήρα
    sensor_width_mm = 13.2
    sensor_height_mm = 8.8
    focal_length_mm = 8.8

    # Επιθυμητό overlap
    frontlap = 0.8
    sidelap = 0.7

    # Περιοχή ενδιαφέροντος (ορθογώνια προσέγγιση)
    area_width_m = 120.0
    area_height_m = 80.0

    # Υπολογισμός αποτυπώματος
    footprint_x, footprint_y = camera_footprint(
        height_m,
        sensor_width_mm,
        sensor_height_mm,
        focal_length_mm,
    )

    print(f"Camera footprint: {footprint_x:.2f} m x {footprint_y:.2f} m")

    # Βήματα λήψεων
    step_x, step_y = compute_step_sizes(
        footprint_x,
        footprint_y,
        frontlap=frontlap,
        sidelap=sidelap,
    )

    print(f"Step X (between lines): {step_x:.2f} m")
    print(f"Step Y (along line):    {step_y:.2f} m")

    # Δημιουργία waypoint grid
    waypoints = generate_grid_waypoints(
        area_width_m,
        area_height_m,
        step_x,
        step_y,
        origin_x=0.0,
        origin_y=0.0,
        boustrophedon=True,
    )

    print(f"Total waypoints: {len(waypoints)}")
    for i, (x, y) in enumerate(waypoints[:10]):
        print(f"WP {i:03d}: x={x:.2f}, y={y:.2f}")
    if len(waypoints) > 10:
        print("... (truncated)")

    # Export σε CSV για χρήση από άλλο module / planner
    with open("waypoints.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x_m", "y_m"])
        for i, (x, y) in enumerate(waypoints):
            writer.writerow([i, x, y])

    print("Waypoints saved to waypoints.csv")


if __name__ == "__main__":
    main()
