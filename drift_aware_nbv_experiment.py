import numpy as np
import matplotlib.pyplot as plt

# Αρχεία εισόδου
COVERAGE_PATH = "coverage_grid_with_uncertainty.csv"
ENTROPY_PATH = "feature_uncertainty_grid.csv"
DRIFT_PATH = "coverage_grid_with_uncertainty_pose.csv"  # pose/drift uncertainty per cell

# Παράμετροι πειράματος
ALPHA = 0.8   # βάρος ποινής drift
TOP_K = 10    # πόσα NBV goals θέλουμε


def load_tabular_grid(path, value_candidates):
    """
    Φορτώνει tabular CSV με στήλες (cell_id, row, col, ..., value_col)
    και το μετατρέπει σε 2D grid [H x W] με βάση (row, col).

    value_candidates: λίστα από υποψήφια ονόματα στήλης για την τιμή του grid.
    """
    print(f"[INFO] Reading tabular grid from {path} ...")
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    names = data.dtype.names
    print("[INFO] Columns:", names)

    if "row" not in names or "col" not in names:
        raise ValueError(
            f"{path} has no 'row'/'col' columns. Available: {names}"
        )

    rows = data["row"].astype(int)
    cols = data["col"].astype(int)

    # Επιλογή στήλης τιμής
    value_col = None
    for cand in value_candidates:
        if cand in names:
            value_col = cand
            break

    if value_col is None:
        raise ValueError(
            f"No suitable value column found in {path}. "
            f"Tried {value_candidates}, available: {names}"
        )

    print(f"[INFO] Using column '{value_col}' as values for {path}")
    vals = data[value_col].astype(float)

    H = rows.max() + 1
    W = cols.max() + 1
    grid = np.zeros((H, W), dtype=float)
    grid[rows, cols] = vals

    print(f"[INFO] Reconstructed grid shape: {grid.shape}")
    return grid


def normalize(grid):
    g = grid.astype(float)
    return (g - g.min()) / (g.max() - g.min() + 1e-8)


def main():
    print("[INFO] Loading grids...")

    # Coverage: 0–1 approximately (πόσο καλυμμένο είναι κάθε κελί)
    cov = load_tabular_grid(
        COVERAGE_PATH,
        value_candidates=["covered", "coverage", "coverage_weight"]
    )

    # Entropy / IG proxy: feature_density_norm ή uncertainty
    entropy = load_tabular_grid(
        ENTROPY_PATH,
        value_candidates=["feature_density_norm", "uncertainty", "entropy"]
    )

    # Drift / pose uncertainty per cell
    drift = load_tabular_grid(
        DRIFT_PATH,
        value_candidates=["pose_uncertainty_norm", "pose_uncertainty", "uncertainty_fused", "uncertainty"]
    )

    if not (cov.shape == entropy.shape == drift.shape):
        raise ValueError(
            f"Grid shapes must match! cov={cov.shape}, entropy={entropy.shape}, drift={drift.shape}"
        )

    print("[INFO] Final grid shape:", cov.shape)

    # Κανονικοποίηση
    entropy_norm = normalize(entropy)
    drift_norm = normalize(drift)
    cov_penalty = cov  # αν δεν είναι 0–1, μπορείς να το normalise αν χρειαστεί

    # Classical NBV: IG * (1 - coverage)
    classical_score = entropy_norm * (1.0 - cov_penalty)

    # Drift-aware NBV: (IG * (1 - coverage)) - α * drift
    drift_score = classical_score - ALPHA * drift_norm

    print("[INFO] Computing rankings...")
    H, W = cov.shape
    flat_idx = np.arange(H * W)

    classical_sorted = flat_idx[np.argsort(classical_score.reshape(-1))[::-1]]
    drift_sorted = flat_idx[np.argsort(drift_score.reshape(-1))[::-1]]

    def idx_to_xy(idx):
        y, x = divmod(idx, W)
        return x, y

    classical_top = [idx_to_xy(i) for i in classical_sorted[:TOP_K]]
    drift_top = [idx_to_xy(i) for i in drift_sorted[:TOP_K]]

    # Αποθήκευση αποτελεσμάτων σε CSV (x,y)
    np.savetxt("nbv_classical_topk.csv", np.array(classical_top), fmt="%d", delimiter=",")
    np.savetxt("nbv_driftaware_topk.csv", np.array(drift_top), fmt="%d", delimiter=",")

    print("[INFO] Saved nbv_classical_topk.csv")
    print("[INFO] Saved nbv_driftaware_topk.csv")

    # ---- Visualization: Classical NBV ----
    plt.figure(figsize=(8, 6))
    plt.imshow(classical_score, origin="lower")
    xs = [p[0] for p in classical_top]
    ys = [p[1] for p in classical_top]
    plt.scatter(xs, ys, marker="o")
    plt.title("Classical NBV (Top K)")
    plt.colorbar(label="Classical score")
    plt.tight_layout()
    plt.savefig("nbv_classical_topk.png")
    print("[INFO] Saved nbv_classical_topk.png")

    # ---- Visualization: Drift-Aware NBV ----
    plt.figure(figsize=(8, 6))
    plt.imshow(drift_score, origin="lower")
    xs = [p[0] for p in drift_top]
    ys = [p[1] for p in drift_top]
    plt.scatter(xs, ys, marker="o")
    plt.title(f"Drift-Aware NBV (Top K, alpha={ALPHA})")
    plt.colorbar(label="Drift-aware score")
    plt.tight_layout()
    plt.savefig("nbv_driftaware_topk.png")
    print("[INFO] Saved nbv_driftaware_topk.png")

    print("\n=== TOP K COMPARISON ===")
    print("Classical:", classical_top)
    print("Drift-aware:", drift_top)


if __name__ == "__main__":
    main()
