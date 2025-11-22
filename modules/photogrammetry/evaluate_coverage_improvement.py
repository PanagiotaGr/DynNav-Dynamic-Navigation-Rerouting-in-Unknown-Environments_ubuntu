import csv
import math
from typing import List, Tuple

import numpy as np

# ΠΡΕΠΕΙ να ταιριάζουν με coverage_map.py
CELL_SIZE = 1.0
COVERAGE_RADIUS = 1.5


def load_coverage_grid(csv_path="coverage_grid.csv"):
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
        raise RuntimeError("coverage_grid.csv empty")

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
    xs, zs = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return xs, zs


def compute_coverage_ratio(covered: np.ndarray) -> float:
    total = covered.size
    covered_cells = int(covered.sum())
    return covered_cells / total if total > 0 else 0.0


def apply_coverage_from_points(
    pts_x: List[float],
    pts_z: List[float],
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    covered: np.ndarray,
    radius: float = COVERAGE_RADIUS,
):
    """
    "Βάφει" επιπλέον κελιά ως καλυμμένα με βάση σημεία (π.χ. replan waypoints).
    Επιστρέφει ΝΕΟ covered array.
    """
    covered_new = covered.copy()
    for x, z in zip(pts_x, pts_z):
        dist2 = (grid_x - x) ** 2 + (grid_z - z) ** 2
        mask = dist2 <= radius**2
        covered_new[mask] = True
    return covered_new


def path_length(xs: List[float], zs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(xs)):
        total += math.dist((xs[i - 1], zs[i - 1]), (xs[i], zs[i]))
    return total


def main():
    # 1. Φόρτωση grid κάλυψης (από VO)
    grid_x, grid_z, covered_before = load_coverage_grid("coverage_grid.csv")
    nz, nx = covered_before.shape
    print(f"Loaded coverage grid: {nx} x {nz} cells")

    # 2. Coverage πριν (VO μόνο)
    cov_before = compute_coverage_ratio(covered_before)
    print(f"Coverage BEFORE replan: {cov_before*100:.2f}%")

    # 3. Φόρτωση replan waypoints
    try:
        rp_xs, rp_zs = load_replan_waypoints("replan_waypoints.csv")
    except FileNotFoundError:
        print("replan_waypoints.csv not found – cannot evaluate improvement.")
        return

    print(f"Loaded {len(rp_xs)} replan waypoints")

    # 4. Εφαρμογή "ιδανικής" κάλυψης replan πάνω στο grid
    covered_after = apply_coverage_from_points(
        rp_xs, rp_zs, grid_x, grid_z, covered_before
    )
    cov_after = compute_coverage_ratio(covered_after)
    print(f"Coverage AFTER replan:  {cov_after*100:.2f}%")

    improvement = cov_after - cov_before
    print(f"Absolute improvement:   {improvement*100:.2f} percentage points")

    # 5. Μήκη διαδρομών & NCE metric
    vo_xs, vo_zs = load_vo_trajectory("vo_trajectory.csv")
    L_vo = path_length(vo_xs, vo_zs)
    L_rp = path_length(rp_xs, rp_zs)
    print(f"VO path length:         {L_vo:.2f}")
    print(f"Replan path length:     {L_rp:.2f}")
    print(f"Total path length:      {L_vo + L_rp:.2f}")

    # Normalized Coverage Efficiency
    if L_vo > 0:
        nce_before = cov_before / L_vo
    else:
        nce_before = 0.0

    if (L_vo + L_rp) > 0:
        nce_after = cov_after / (L_vo + L_rp)
    else:
        nce_after = 0.0

    print("=== Normalized Coverage Efficiency (coverage per unit path) ===")
    print(f"NCE BEFORE: {nce_before:.6f}")
    print(f"NCE AFTER:  {nce_after:.6f}")


if __name__ == "__main__":
    main()
