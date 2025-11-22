import csv
import numpy as np
import matplotlib.pyplot as plt


def load_coverage_grid(csv_path="coverage_grid.csv"):
    """
    Φορτώνει το coverage_grid.csv και αναδομεί:
      - πίνακα covered (nz x nx)
      - πίνακα center_x (nz x nx)
      - πίνακα center_z (nz x nx)
    Χρησιμοποιεί τα row/col που έχουμε ήδη αποθηκεύσει.
    """
    rows = []
    cols = []
    center_xs = []
    center_zs = []
    covered_vals = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["row"])
            c = int(row["col"])
            cx = float(row["center_x"])
            cz = float(row["center_z"])
            covered = bool(int(row["covered"]))

            rows.append(r)
            cols.append(c)
            center_xs.append(cx)
            center_zs.append(cz)
            covered_vals.append(covered)

    if not rows:
        raise RuntimeError("coverage_grid.csv is empty or not found")

    max_r = max(rows)
    max_c = max(cols)

    nz = max_r + 1
    nx = max_c + 1

    grid_x = np.zeros((nz, nx), dtype=float)
    grid_z = np.zeros((nz, nx), dtype=float)
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


def load_replan_waypoints(csv_path="replan_waypoints.csv"):
    """
    Φορτώνει τα replan_waypoints.csv.
    Περιμένουμε στήλες: id, x, z (DictReader -> keys: "id", "x", "z").
    """
    xs, zs = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return xs, zs


def main():
    # 1. Φόρτωση coverage grid
    grid_x, grid_z, covered = load_coverage_grid("coverage_grid.csv")
    nz, nx = covered.shape
    print(f"Loaded coverage grid: {nx} x {nz} cells")

    # 2. Φόρτωση VO τροχιάς
    vo_xs, vo_zs = load_vo_trajectory("vo_trajectory.csv")
    print(f"Loaded VO trajectory with {len(vo_xs)} poses")

    # 3. Φόρτωση replan waypoints (αν υπάρχουν)
    try:
        rp_xs, rp_zs = load_replan_waypoints("replan_waypoints.csv")
        print(f"Loaded {len(rp_xs)} replan waypoints")
        has_replan = len(rp_xs) > 0
    except FileNotFoundError:
        print("replan_waypoints.csv not found, will plot only coverage + VO.")
        rp_xs, rp_zs = [], []
        has_replan = False

    # 4. Δημιουργία heatmap κάλυψης
    data = covered.astype(int)

    # Υπολογισμός extent για imshow
    xmin = grid_x[0, 0]
    xmax = grid_x[0, -1]
    zmin = grid_z[0, 0]
    zmax = grid_z[-1, 0]

    # Εκτίμηση cell_size από διαφορά γειτονικών centers
    if nx > 1:
        cell_size_x = abs(grid_x[0, 1] - grid_x[0, 0])
    else:
        cell_size_x = 1.0
    if nz > 1:
        cell_size_z = abs(grid_z[1, 0] - grid_z[0, 0])
    else:
        cell_size_z = 1.0

    cell_size = max(cell_size_x, cell_size_z)

    extent = [
        xmin - 0.5 * cell_size,
        xmax + 0.5 * cell_size,
        zmin - 0.5 * cell_size,
        zmax + 0.5 * cell_size,
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

    # 5. VO τροχιά
    plt.plot(vo_xs, vo_zs, "k-", linewidth=1.0, label="VO trajectory")

    # 6. Replan waypoints (αν υπάρχουν)
    if has_replan:
        plt.scatter(rp_xs, rp_zs, marker="x", s=40, label="Replan waypoints")

    plt.xlabel("x (VO units)")
    plt.ylabel("z (VO units)")
    plt.title("Coverage + VO trajectory + Replan waypoints")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
