import csv
import matplotlib.pyplot as plt


def load_waypoints(csv_path="waypoints_poly.csv"):
    xs = []
    ys = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x_m"]))
            ys.append(float(row["y_m"]))
    return xs, ys


def main():
    # Φόρτωση waypoints από CSV
    xs, ys = load_waypoints("waypoints_poly.csv")
    print(f"Loaded {len(xs)} waypoints from waypoints_poly.csv")

    # Ορισμός AOI polygon (ίδιο με path_planner_poly)
    aoi_x = [0, 120, 120, 0, 0]
    aoi_y = [0, 0, 80, 80, 0]

    # No-fly ζώνη (ίδια με path_planner_poly)
    nf_x = [40, 80, 80, 40, 40]
    nf_y = [30, 30, 50, 50, 30]

    plt.figure()

    # AOI boundary
    plt.plot(aoi_x, aoi_y, linestyle="-", label="AOI boundary")

    # No-fly zone boundary
    plt.plot(nf_x, nf_y, linestyle="--", label="No-fly zone")

    # Waypoints (filtered)
    plt.scatter(xs, ys, s=30, marker="o", label="Waypoints (valid)")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Coverage waypoints with AOI and no-fly zone")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # για να μην παραμορφωθεί το σχήμα

    plt.show()


if __name__ == "__main__":
    main()
