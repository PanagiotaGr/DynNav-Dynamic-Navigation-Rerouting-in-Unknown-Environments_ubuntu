#coverage (0/1)

#VO trajectory

#weighted replan waypoints (διαφορετικό χρώμα από τα απλά replan)

import csv
import numpy as np
import matplotlib.pyplot as plt


def load_coverage_grid(csv_path="coverage_grid.csv"):
    rows = []
    cols = []
    center_xs = []
    center_zs = []
    covered_vals = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(int(row["row"]))
            cols.append(int(row["col"]))
            center_xs.append(float(row["center_x"]))
            center_zs.append(float(row["center_z"]))
            covered_vals.append(bool(int(row["covered"])))

    nz = max(rows) + 1
    nx = max(cols) + 1

    grid_x = np.zeros((nz, nx))
    grid_z = np.zeros((nz, nx))
    covered = np.zeros((nz, nx), dtype=bool)

    for r, c, cx, cz, cov in zip(rows, cols, center_xs, center_zs, covered_vals):
        grid_x[r, c] = cx
        grid_z[r, c] = cz
        covered[r, c] = cov

    return grid_x, grid_z, covered


def load_vo_trajectory(csv_path="vo_trajectory.csv"):
    xs, zs = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return xs, zs


def load_weighted_replan(csv_path="replan_weighted_waypoints.csv"):
    xs, zs = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # id, x, z
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return xs, zs


def main():
    grid_x, grid_z, covered = load_coverage_grid("coverage_grid.csv")
    vo_xs, vo_zs = load_vo_trajectory("vo_trajectory.csv")
    rp_xs, rp_zs = load_weighted_replan("replan_weighted_waypoints.csv")

    nz, nx = covered.shape
    data = covered.astype(int)

    xmin = grid_x[0, 0]
    xmax = grid_x[0, -1]
    zmin = grid_z[0, 0]
    zmax = grid_z[-1, 0]

    cell_x = abs(grid_x[0, 1] - grid_x[0, 0]) if nx > 1 else 1.0
    cell_z = abs(grid_z[1, 0] - grid_z[0, 0]) if nz > 1 else 1.0
    cell = max(cell_x, cell_z)

    extent = [
        xmin - 0.5 * cell,
        xmax + 0.5 * cell,
        zmin - 0.5 * cell,
        zmax + 0.5 * cell,
    ]

    plt.figure()
    plt.imshow(
        data,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(label="Coverage (0/1)")

    plt.plot(vo_xs, vo_zs, "k-", linewidth=1.0, label="VO trajectory")

    if rp_xs:
        plt.plot(rp_xs, rp_zs, "-", linewidth=1.0, label="Weighted replan path")
        plt.scatter(rp_xs, rp_zs, marker="x", s=30)

    plt.xlabel("x (VO units)")
    plt.ylabel("z (VO units)")
    plt.title("Coverage + VO + Weighted Replan")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
