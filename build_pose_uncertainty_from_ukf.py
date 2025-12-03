import numpy as np
import pandas as pd


def main():
    traj_path = "ukf_fused_trajectory.csv"
    coverage_path = "coverage_grid_with_uncertainty.csv"
    output_path = "coverage_grid_with_uncertainty_pose.csv"

    print(f"[INFO] Loading fused trajectory from: {traj_path}")
    traj_df = pd.read_csv(traj_path)

    required_traj_cols = ["x_vo", "y_vo", "x_ukf", "y_ukf"]
    for c in required_traj_cols:
        if c not in traj_df.columns:
            raise ValueError(
                f"Column '{c}' not found in {traj_path}. "
                f"Available columns: {list(traj_df.columns)}"
            )

    # Υπολογισμός drift VO–UKF για κάθε timestep
    dx = traj_df["x_vo"].to_numpy() - traj_df["x_ukf"].to_numpy()
    dy = traj_df["y_vo"].to_numpy() - traj_df["y_ukf"].to_numpy()
    drift = np.sqrt(dx ** 2 + dy ** 2)

    print("[INFO] Drift stats (VO vs UKF):")
    print(f"       min = {drift.min():.4f}, max = {drift.max():.4f}, mean = {drift.mean():.4f}")

    # Φορτώνουμε coverage grid
    print(f"[INFO] Loading coverage grid from: {coverage_path}")
    cov_df = pd.read_csv(coverage_path)

    # Ορίζουμε ρητά ποιες στήλες είναι οι συντεταγμένες κελιού στο χώρο
    x_col = "center_x"
    y_col = "center_z"
    if x_col not in cov_df.columns or y_col not in cov_df.columns:
        raise ValueError(
            f"Coverage grid does not contain required columns '{x_col}', '{y_col}'. "
            f"Available columns: {list(cov_df.columns)}"
        )

    print(f"[INFO] Using coverage coordinate columns: x='{x_col}', y='{y_col}'")

    cell_coords = cov_df[[x_col, y_col]].to_numpy()  # (N_cells, 2)
    n_cells = cell_coords.shape[0]

    # Συντεταγμένες των fused poses (χρησιμοποιούμε UKF pose)
    pose_coords = np.stack(
        [traj_df["x_ukf"].to_numpy(), traj_df["y_ukf"].to_numpy()],
        axis=1
    )  # (N_poses, 2)
    n_poses = pose_coords.shape[0]

    print(f"[INFO] Number of poses: {n_poses}, number of cells: {n_cells}")

    # Για κάθε pose, βρίσκουμε το κοντινότερο κελί (nearest neighbor σε 2D)
    # Απλή O(N_poses * N_cells) υλοποίηση (επαρκής για μεσαία grids)
    cell_indices = np.empty(n_poses, dtype=int)

    for k in range(n_poses):
        p = pose_coords[k]  # (2,)
        diffs = cell_coords - p[None, :]          # (N_cells, 2)
        d2 = np.sum(diffs ** 2, axis=1)           # (N_cells,)
        cell_indices[k] = int(np.argmin(d2))

    # Τώρα για κάθε κελί μαζεύουμε drift από τα poses που αντιστοιχούν εκεί
    sum_drift = np.zeros(n_cells, dtype=float)
    count = np.zeros(n_cells, dtype=int)

    for k in range(n_poses):
        ci = cell_indices[k]
        sum_drift[ci] += drift[k]
        count[ci] += 1

    pose_uncertainty = np.zeros(n_cells, dtype=float)
    non_empty = count > 0
    pose_uncertainty[non_empty] = sum_drift[non_empty] / count[non_empty]

    print("[INFO] Pose-uncertainty per cell stats (non-empty cells only):")
    if np.any(non_empty):
        print(f"       min = {pose_uncertainty[non_empty].min():.4f}, "
              f"max = {pose_uncertainty[non_empty].max():.4f}, "
              f"mean = {pose_uncertainty[non_empty].mean():.4f}")
    else:
        print("       (no cells received any poses!)")

    # Κανονικοποίηση στο [0, 1]
    pose_uncertainty_norm = np.zeros_like(pose_uncertainty)
    if np.any(non_empty):
        u_min = pose_uncertainty[non_empty].min()
        u_max = pose_uncertainty[non_empty].max()
        if u_max > u_min:
            pose_uncertainty_norm[non_empty] = (
                (pose_uncertainty[non_empty] - u_min) / (u_max - u_min)
            )
        else:
            pose_uncertainty_norm[non_empty] = 0.0

    # Προσθέτουμε τις στήλες στο coverage grid
    cov_df["pose_uncertainty"] = pose_uncertainty
    cov_df["pose_uncertainty_norm"] = pose_uncertainty_norm

    # Αν υπάρχει ήδη στήλη 'uncertainty', φτιάχνουμε fused έκδοση
    if "uncertainty" in cov_df.columns:
        cov_df["uncertainty_fused"] = (
            0.5 * cov_df["uncertainty"].to_numpy() +
            0.5 * cov_df["pose_uncertainty_norm"].to_numpy()
        )
        print("[INFO] Created 'uncertainty_fused' column combining old uncertainty and pose-based.")

    # Σώζουμε σε νέο αρχείο
    cov_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved updated coverage grid with pose uncertainty to: {output_path}")


if __name__ == "__main__":
    main()
