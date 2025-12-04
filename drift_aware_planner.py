import numpy as np
import pandas as pd
import math

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
    load_cluster_centroids,
    information_gain,
)


def load_drift_model(model_path="drift_model.npz"):
    """
    Φορτώνει το ridge μοντέλο από το train_drift_predictor.py
    Περιμένει: w, b, mean, std, feature_cols
    """
    data = np.load(model_path, allow_pickle=True)
    w = data["w"]
    b = float(data["b"])
    mean = data["mean"]
    std = data["std"]
    feature_cols = data["feature_cols"]
    return w, b, mean, std, feature_cols


def predict_drift(entropy, local_uncertainty, speed, w, b, mean, std):
    """
    Υπολογίζει predicted drift για ένα σημείο.
    Features: [entropy, local_uncertainty, speed]
    """
    x = np.array([entropy, local_uncertainty, speed], dtype=float)
    x_norm = (x - mean) / std
    y_pred = float(x_norm @ w + b)
    return max(y_pred, 0.0)  # δεν μας νοιάζει αρνητικός drift → clamp στο 0


if __name__ == "__main__":
    # Μπορείς να βάλεις και το VLA-adjusted coverage εδώ:
    coverage_csv = "coverage_grid_with_uncertainty_vla.csv"
    centroids_csv = "cluster_waypoints.csv"
    drift_model_path = "drift_model.npz"
    output_csv = "drift_aware_candidates.csv"

    robot_row, robot_col = 0, 0
    window_radius = 4
    alpha = 5.0  # πόσο δυνατά τιμωρούμε το predicted drift

    print(f"[DRIFT-PLANNER] Loading coverage grid from: {coverage_csv}")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)
    U_map = load_uncertainty_grid(coverage_csv)

    print(f"[DRIFT-PLANNER] Loading cluster centroids from: {centroids_csv}")
    centroids_df = load_cluster_centroids(centroids_csv)

    print(f"[DRIFT-PLANNER] Loading drift model from: {drift_model_path}")
    w, b, mean, std, feature_cols = load_drift_model(drift_model_path)
    print(f"[DRIFT-PLANNER] Drift model feature order: {feature_cols}")

    candidates = []

    for _, r in centroids_df.iterrows():
        cx = float(r["x"])
        cy = float(r["y"])

        # Grid indices
        crow = int(round(cy))
        ccol = int(round(cx))

        # 1) IG γύρω από το cluster
        ig = information_gain(H_map, crow, ccol, window_radius=window_radius)

        # 2) Απόσταση από ρομπότ
        dist = math.sqrt((crow - robot_row) ** 2 + (ccol - robot_col) ** 2) + 1e-3

        # 3) Τοπική entropy / uncertainty
        if 0 <= crow < H_map.shape[0] and 0 <= ccol < H_map.shape[1]:
            local_H = float(H_map[crow, ccol])
            local_U = float(U_map[crow, ccol])
        else:
            local_H = 0.0
            local_U = 0.0

        # 4) proxy για speed: εδώ το βάζουμε = dist (ή μπορείς = 1.0)
        speed_proxy = dist

        # 5) Predicted drift από το μοντέλο
        pred_drift = predict_drift(local_H, local_U, speed_proxy, w, b, mean, std)

        # 6) Βασικός score (IG / distance)
        base_score = ig / dist

        # 7) Drift-aware score
        final_score = base_score / (1.0 + alpha * pred_drift)

        candidates.append({
            "x": cx,
            "y": cy,
            "row": crow,
            "col": ccol,
            "IG": ig,
            "distance": dist,
            "local_entropy": local_H,
            "local_uncertainty": local_U,
            "speed_proxy": speed_proxy,
            "predicted_drift": pred_drift,
            "base_score": base_score,
            "final_score": final_score,
        })

    cand_df = pd.DataFrame(candidates)
    cand_df.sort_values(by="final_score", ascending=False, inplace=True)
    cand_df.to_csv(output_csv, index=False)
    print(f"[DRIFT-PLANNER] Saved drift-aware candidates to: {output_csv}")

    if len(cand_df) == 0:
        print("[DRIFT-PLANNER] No candidates found.")
    else:
        best = cand_df.iloc[0]
        print("\n[DRIFT-PLANNER] === Best drift-aware goal ===")
        print(best[[
            "x", "y", "IG", "distance",
            "local_entropy", "local_uncertainty",
            "predicted_drift", "base_score", "final_score"
        ]])
        print("===========================================\n")

        # Optionally: save μόνο το best σε ξεχωριστό CSV
        best_goal_df = pd.DataFrame([{
            "goal_x": best["x"],
            "goal_y": best["y"],
            "goal_row": best["row"],
            "goal_col": best["col"],
            "predicted_drift": best["predicted_drift"],
            "final_score": best["final_score"],
        }])
        best_goal_df.to_csv("drift_aware_goal.csv", index=False)
        print("[DRIFT-PLANNER] Saved best drift-aware goal to: drift_aware_goal.csv")
