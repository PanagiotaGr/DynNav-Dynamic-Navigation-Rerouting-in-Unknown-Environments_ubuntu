import csv
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ====== ΠΑΡΑΜΕΤΡΟΙ ΠΕΙΡΑΜΑΤΟΣ ======
CELL_SIZE = 1.0          # μέγεθος κελιού grid (σε ίδιες μονάδες με VO, π.χ. "μέτρα")
COVERAGE_RADIUS = 1.5    # ακτίνα κάλυψης γύρω από κάθε pose
MARGIN_RATIO = 0.1       # επιπλέον περιθώριο γύρω από την τροχιά


def load_trajectory(csv_path: str = "vo_trajectory.csv"):
    """Φορτώνει την τροχιά από το vo_trajectory.csv."""
    frames = []
    xs, ys, zs = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame_idx"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
    return frames, xs, ys, zs


def build_grid_from_trajectory(
    xs: List[float],
    zs: List[float],
    cell_size: float = CELL_SIZE,
    margin_ratio: float = MARGIN_RATIO,
):
    """
    Ορίζει grid (AOI) με βάση το bounding box της τροχιάς VO.
    Επιστρέφει:
      - xs, zs των centers των κελιών σε 2D πίνακες
      - covered: πίνακας boolean αρχικά False
    """
    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)

    dx = xmax - xmin
    dz = zmax - zmin

    if dx == 0:
        dx = 1.0
    if dz == 0:
        dz = 1.0

    xmin -= margin_ratio * dx
    xmax += margin_ratio * dx
    zmin -= margin_ratio * dz
    zmax += margin_ratio * dz

    # αριθμός κελιών σε κάθε διάσταση
    nx = int(math.ceil((xmax - xmin) / cell_size))
    nz = int(math.ceil((zmax - zmin) / cell_size))

    # centers κελιών
    xs_centers = xmin + (np.arange(nx) + 0.5) * cell_size
    zs_centers = zmin + (np.arange(nz) + 0.5) * cell_size

    # δημιουργία 2D πλέγματος (grid)
    grid_x, grid_z = np.meshgrid(xs_centers, zs_centers)  # shape (nz, nx)

    covered = np.zeros((nz, nx), dtype=bool)

    bounds = {
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "nx": nx,
        "nz": nz,
        "cell_size": cell_size,
    }

    return grid_x, grid_z, covered, bounds


def mark_coverage_from_trajectory(
    xs: List[float],
    zs: List[float],
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    covered: np.ndarray,
    coverage_radius: float = COVERAGE_RADIUS,
):
    """
    Για κάθε pose της VO, μαρκάρει ως καλυμμένα τα κελιά του grid
    των οποίων το κέντρο βρίσκεται εντός ακτίνας coverage_radius.
    """

    nz, nx = covered.shape

    # προϋπολογισμός: για κάθε pose, ελέγχουμε όλα τα κελιά (O(Ncells * Nposes)).
    # Για experimental scale αυτό είναι ΟΚ.
    for i, (x, z) in enumerate(zip(xs, zs)):
        # αποφυγή πολύ βαριάς εκτύπωσης
        # if i % 20 == 0:
        #     print(f"Processing pose {i}/{len(xs)}")

        # απόσταση από κάθε κέντρο κελιού
        dist2 = (grid_x - x) ** 2 + (grid_z - z) ** 2
        mask = dist2 <= coverage_radius ** 2

        covered[mask] = True

    return covered


def compute_coverage_statistics(covered: np.ndarray, bounds: dict):
    total_cells = covered.size
    covered_cells = int(covered.sum())
    ratio = covered_cells / total_cells if total_cells > 0 else 0.0

    print("=== Coverage statistics ===")
    print(f"Grid size: {bounds['nx']} x {bounds['nz']} cells")
    print(f"Total cells:   {total_cells}")
    print(f"Covered cells: {covered_cells}")
    print(f"Coverage:      {ratio * 100:.1f}%")

    return {
        "total_cells": total_cells,
        "covered_cells": covered_cells,
        "coverage_ratio": ratio,
    }


def save_coverage_grid_csv(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    covered: np.ndarray,
    csv_path: str = "coverage_grid.csv",
):
    """
    Αποθήκευση του grid σε CSV (κάθε κελιά με center_x, center_z, covered).
    """
    nz, nx = covered.shape
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "row", "col", "center_x", "center_z", "covered"])
        cell_id = 0
        for r in range(nz):
            for c in range(nx):
                writer.writerow(
                    [
                        cell_id,
                        r,
                        c,
                        float(grid_x[r, c]),
                        float(grid_z[r, c]),
                        int(covered[r, c]),
                    ]
                )
                cell_id += 1

    print(f"Saved coverage grid to {csv_path}")


def plot_coverage_heatmap(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    covered: np.ndarray,
    xs_traj: List[float],
    zs_traj: List[float],
):
    """
    Σχεδιάζει heatmap κάλυψης και πάνω την τροχιά της VO.
    """
    nz, nx = covered.shape

    # φτιάχνουμε ένα matrix 0/1 για imshow
    data = covered.astype(int)

    plt.figure()
    # extent = [xmin, xmax, zmin, zmax] -> για να ταιριάξουν οι άξονες με coords
    xmin = grid_x[0, 0] - 0.5 * CELL_SIZE
    xmax = grid_x[0, -1] + 0.5 * CELL_SIZE
    zmin = grid_z[0, 0] - 0.5 * CELL_SIZE
    zmax = grid_z[-1, 0] + 0.5 * CELL_SIZE

    plt.imshow(
        data,
        origin="lower",
        extent=[xmin, xmax, zmin, zmax],
        interpolation="nearest",
        aspect="equal",
    )

    plt.colorbar(label="Coverage (0/1)")
    plt.plot(xs_traj, zs_traj, "k-", linewidth=1.0, label="VO trajectory")

    plt.xlabel("x (VO units)")
    plt.ylabel("z (VO units)")
    plt.title("Coverage map from VO trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # 1. Φόρτωση τροχιάς από VO
    frames, xs, ys, zs = load_trajectory("vo_trajectory.csv")
    print(f"Loaded {len(frames)} poses from vo_trajectory.csv")

    if len(xs) == 0:
        print("No poses found, aborting.")
        return

    # 2. Δημιουργία grid γύρω από την τροχιά (x-z επίπεδο)
    grid_x, grid_z, covered, bounds = build_grid_from_trajectory(xs, zs)

    # 3. Μαρκάρισμα κάλυψης με βάση ακτίνα κάλυψης
    covered = mark_coverage_from_trajectory(xs, zs, grid_x, grid_z, covered)

    # 4. Στατιστικά
    stats = compute_coverage_statistics(covered, bounds)

    # 5. Αποθήκευση σε CSV
    save_coverage_grid_csv(grid_x, grid_z, covered, "coverage_grid.csv")

    # 6. Plot / heatmap
    plot_coverage_heatmap(grid_x, grid_z, covered, xs, zs)


if __name__ == "__main__":
    main()
