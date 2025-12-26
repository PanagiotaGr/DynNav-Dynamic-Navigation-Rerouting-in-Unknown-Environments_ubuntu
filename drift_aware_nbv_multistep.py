import numpy as np
import matplotlib.pyplot as plt

# Αρχεία εισόδου (όπως πριν)
COVERAGE_PATH = "coverage_grid_with_uncertainty.csv"
ENTROPY_PATH = "feature_uncertainty_grid.csv"
DRIFT_PATH = "coverage_grid_with_uncertainty_pose.csv"

# Παράμετροι πειράματος
STEPS = 30          # πόσα NBV βήματα θα κάνουμε ανά policy
ALPHA_DRIFT = 0.8   # βάρος drift penalty για drift-aware policy
RANDOM_TIE_BREAK = True  # αν True, σπάει ισοπαλίες τυχαία


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


def run_episode(cov_init, entropy_init, drift_init, alpha, steps, policy_name):
    """
    Τρέχει ένα multi-step NBV επεισόδιο.
    alpha = 0.0  -> classical
    alpha > 0.0  -> drift-aware

    Επιστρέφει:
    - λίστα (x, y) στόχων
    - IG_per_step
    - drift_per_step
    """
    cov = cov_init.copy()
    entropy = entropy_init.copy()
    drift = drift_init.copy()

    H, W = cov.shape

    entropy_norm = normalize(entropy)
    drift_norm = normalize(drift)

    goals = []
    ig_per_step = []
    drift_per_step = []

    for t in range(steps):
        # ενημέρωση scores για κάθε βήμα με βάση current coverage
        classical_score = entropy_norm * (1.0 - cov)
        score = classical_score - alpha * drift_norm

        flat = score.reshape(-1)
        if RANDOM_TIE_BREAK:
            # μικρό jitter για να σπάσει ισοπαλίες
            flat = flat + 1e-6 * np.random.randn(flat.size)

        best_idx = int(np.argmax(flat))
        y, x = divmod(best_idx, W)

        # IG contribution = classical_score στο συγκεκριμένο κελί
        ig_value = float(classical_score[y, x])
        drift_value = float(drift_norm[y, x])

        goals.append((x, y))
        ig_per_step.append(ig_value)
        drift_per_step.append(drift_value)

        # ενημέρωση coverage & entropy:
        # θεωρούμε ότι αυτό το κελί καλύφθηκε πλήρως -> coverage=1, entropy=0
        cov[y, x] = 1.0
        entropy_norm[y, x] = 0.0  # δεν προσφέρει άλλο IG

        print(
            f"[{policy_name}] step {t+1}/{steps}: goal=({x},{y}), "
            f"IG={ig_value:.4f}, drift={drift_value:.4f}"
        )

    return goals, np.array(ig_per_step), np.array(drift_per_step)


def main():
    print("[INFO] Loading grids for multi-step experiment...")

    cov = load_tabular_grid(
        COVERAGE_PATH,
        value_candidates=["covered", "coverage", "coverage_weight"]
    )
    entropy = load_tabular_grid(
        ENTROPY_PATH,
        value_candidates=["feature_density_norm", "uncertainty", "entropy"]
    )
    drift = load_tabular_grid(
        DRIFT_PATH,
        value_candidates=["pose_uncertainty_norm", "pose_uncertainty", "uncertainty_fused", "uncertainty"]
    )

    if not (cov.shape == entropy.shape == drift.shape):
        raise ValueError(
            f"Grid shapes must match! cov={cov.shape}, entropy={entropy.shape}, drift={drift.shape}"
        )

    print("[INFO] Final grid shape:", cov.shape)

    # Episode 1: Classical (alpha = 0)
    goals_classical, ig_classical, drift_classical = run_episode(
        cov, entropy, drift, alpha=0.0, steps=STEPS, policy_name="CLASSICAL"
    )

    # Episode 2: Drift-aware (alpha = ALPHA_DRIFT)
    goals_drift, ig_drift, drift_drift = run_episode(
        cov, entropy, drift, alpha=ALPHA_DRIFT, steps=STEPS, policy_name="DRIFT-AWARE"
    )

    # Save metrics to CSV
    steps_idx = np.arange(1, STEPS + 1)

    classical_metrics = np.column_stack(
        [steps_idx, ig_classical, drift_classical]
    )
    drift_metrics = np.column_stack(
        [steps_idx, ig_drift, drift_drift]
    )

    np.savetxt(
        "nbv_multistep_classical_metrics.csv",
        classical_metrics,
        delimiter=",",
        header="step,ig,drift",
        comments="",
    )
    np.savetxt(
        "nbv_multistep_driftaware_metrics.csv",
        drift_metrics,
        delimiter=",",
        header="step,ig,drift",
        comments="",
    )

    print("[INFO] Saved nbv_multistep_classical_metrics.csv")
    print("[INFO] Saved nbv_multistep_driftaware_metrics.csv")

    # Save goals
    np.savetxt(
        "nbv_multistep_classical_goals.csv",
        np.array(goals_classical),
        fmt="%d",
        delimiter=",",
        header="x,y",
        comments="",
    )
    np.savetxt(
        "nbv_multistep_driftaware_goals.csv",
        np.array(goals_drift),
        fmt="%d",
        delimiter=",",
        header="x,y",
        comments="",
    )

    print("[INFO] Saved nbv_multistep_*_goals.csv")

    # ---- Plot IG per step ----
    plt.figure()
    plt.plot(steps_idx, ig_classical, marker="o", label="Classical")
    plt.plot(steps_idx, ig_drift, marker="o", label=f"Drift-aware (alpha={ALPHA_DRIFT})")
    plt.xlabel("Step")
    plt.ylabel("Information Gain proxy")
    plt.title("IG per NBV step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nbv_multistep_ig.png")
    print("[INFO] Saved nbv_multistep_ig.png")

    # ---- Plot drift per step ----
    plt.figure()
    plt.plot(steps_idx, drift_classical, marker="o", label="Classical")
    plt.plot(steps_idx, drift_drift, marker="o", label=f"Drift-aware (alpha={ALPHA_DRIFT})")
    plt.xlabel("Step")
    plt.ylabel("Normalized drift exposure")
    plt.title("Drift per NBV step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nbv_multistep_drift.png")
    print("[INFO] Saved nbv_multistep_drift.png")

    # ---- Print summary ----
    print("\n=== SUMMARY ===")
    print(f"Classical: total IG = {ig_classical.sum():.4f}, total drift = {drift_classical.sum():.4f}")
    print(f"Drift-aware (alpha={ALPHA_DRIFT}): total IG = {ig_drift.sum():.4f}, total drift = {drift_drift.sum():.4f}")


if __name__ == "__main__":
    main()
