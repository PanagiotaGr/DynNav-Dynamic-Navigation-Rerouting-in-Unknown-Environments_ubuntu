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
    frames = []
    xs, zs = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame_idx"]))
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return frames, xs, zs


def load_vo_stats(csv_path="vo_stats.csv"):
    stats = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fidx = int(row["frame_idx"])
            num_inliers = int(row["num_inliers"])
            stats[fidx] = num_inliers
    return stats


def build_feature_density_map(grid_x, grid_z, frames, xs, zs, stats_dict):
    """
    Για κάθε pose της VO, βρίσκουμε το πιο κοντινό κελί του grid
    και αθροίζουμε num_inliers εκεί. Στο τέλος κάνουμε average.
    """
    nz, nx = grid_x.shape
    feature_sum = np.zeros((nz, nx), dtype=float)
    counts = np.zeros((nz, nx), dtype=int)

    for frame_idx, x, z in zip(frames, xs, zs):
        num_inliers = stats_dict.get(frame_idx, 0)

        # αν δεν έχει καθόλου inliers, το skip-άρουμε ή το μετράμε ως 0
        # εδώ: απλά το καταγράφουμε σαν 0 αν τραβήξει nearest cell
        # (counts++ με value 0 δεν αλλάζει feature_sum)
        dist2 = (grid_x - x) ** 2 + (grid_z - z) ** 2
        r, c = np.unravel_index(np.argmin(dist2), dist2.shape)

        feature_sum[r, c] += num_inliers
        counts[r, c] += 1

    # feature_density = μέσος αριθμός inliers ανά επίσκεψη
    feature_density = np.zeros((nz, nx), dtype=float)
    nonzero_mask = counts > 0
    feature_density[nonzero_mask] = feature_sum[nonzero_mask] / counts[nonzero_mask]

    return feature_density, counts


def compute_uncertainty(feature_density):
    """
    Κανονικοποίηση σε [0,1] και υπολογισμός uncertainty = 1 - density_norm.
    Για κελιά που δεν επισκέφθηκαν ποτέ (density=0 & counts=0) θα το ορίσουμε στο 1.
    """
    nz, nx = feature_density.shape
    fd = feature_density.copy()
    max_val = fd.max()

    if max_val > 0:
        fd_norm = fd / max_val
    else:
        fd_norm = np.zeros_like(fd)

    uncertainty = 1.0 - fd_norm
    return fd_norm, uncertainty


def save_feature_uncertainty_grid(
    grid_x, grid_z, feature_density_norm, uncertainty, csv_path="feature_uncertainty_grid.csv"
):
    nz, nx = grid_x.shape
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["cell_id", "row", "col", "center_x", "center_z", "feature_density_norm", "uncertainty"]
        )
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
                        float(feature_density_norm[r, c]),
                        float(uncertainty[r, c]),
                    ]
                )
                cell_id += 1
    print(f"Saved feature/uncertainty grid to {csv_path}")


def plot_maps(grid_x, grid_z, feature_density_norm, uncertainty):
    nz, nx = grid_x.shape

    xmin = grid_x[0, 0]
    xmax = grid_x[0, -1]
    zmin = grid_z[0, 0]
    zmax = grid_z[-1, 0]

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
        feature_density_norm,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(label="Normalized feature density")
    plt.title("VO feature-density map")
    plt.xlabel("x (VO units)")
    plt.ylabel("z (VO units)")
    plt.grid(True)

    plt.figure()
    plt.imshow(
        uncertainty,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(label="Uncertainty (1 - density)")
    plt.title("Uncertainty map from VO feature density")
    plt.xlabel("x (VO units)")
    plt.ylabel("z (VO units)")
    plt.grid(True)

    plt.show()


def main():
    # 1. Grid & κάλυψη
    grid_x, grid_z, covered = load_coverage_grid("coverage_grid.csv")
    nz, nx = grid_x.shape
    print(f"Loaded coverage grid: {nx} x {nz} cells")

    # 2. VO trajectory
    frames, xs, zs = load_vo_trajectory("vo_trajectory.csv")
    print(f"Loaded VO poses: {len(frames)}")

    # 3. VO stats (inliers per frame)
    stats_dict = load_vo_stats("vo_stats.csv")
    print(f"Loaded VO stats for {len(stats_dict)} frames")

    # 4. Feature-density map
    feature_density, counts = build_feature_density_map(grid_x, grid_z, frames, xs, zs, stats_dict)
    print("Computed feature density map.")

    # 5. Uncertainty map
    feature_density_norm, uncertainty = compute_uncertainty(feature_density)
    print("Computed normalized density and uncertainty map.")

    # 6. Save σε CSV
    save_feature_uncertainty_grid(grid_x, grid_z, feature_density_norm, uncertainty)

    # 7. Plot
    plot_maps(grid_x, grid_z, feature_density_norm, uncertainty)


if __name__ == "__main__":
    main()
