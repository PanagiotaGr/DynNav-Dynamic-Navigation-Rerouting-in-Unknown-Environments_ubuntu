import numpy as np
import pandas as pd
import math

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map
)


def compute_vo_drift(vo_df):
    """
    Υπολογίζει VO drift ανά frame.
    Drift = ||Δpose(i) - Δpose(i-1)||, όπως σε SLAM papers.
    """
    drifts = [0.0]
    for i in range(1, len(vo_df)):
        dx1 = vo_df.loc[i, "x"] - vo_df.loc[i - 1, "x"]
        dy1 = vo_df.loc[i, "y"] - vo_df.loc[i - 1, "y"]

        dx0 = vo_df.loc[i - 1, "x"] - vo_df.loc[i - 2, "x"] if i >= 2 else 0
        dy0 = vo_df.loc[i - 1, "y"] - vo_df.loc[i - 2, "y"] if i >= 2 else 0

        drift = math.sqrt((dx1 - dx0) ** 2 + (dy1 - dy0) ** 2)
        drifts.append(drift)

    return drifts


def extract_local_entropy(H_map, x, y):
    r = int(round(y))
    c = int(round(x))
    if 0 <= r < H_map.shape[0] and 0 <= c < H_map.shape[1]:
        return float(H_map[r, c])
    return 0.0


def extract_local_uncertainty(U_map, x, y):
    r = int(round(y))
    c = int(round(x))
    if 0 <= r < U_map.shape[0] and 0 <= c < U_map.shape[1]:
        return float(U_map[r, c])
    return 0.0


if __name__ == "__main__":

    vo_csv = "vo_trajectory.csv"
    coverage_csv = "coverage_grid_with_uncertainty.csv"
    output_csv = "drift_dataset.csv"

    print("[DATASET] Loading VO trajectory...")
    vo_df = pd.read_csv(vo_csv)

    print("[DATASET] Computing VO drift...")
    drifts = compute_vo_drift(vo_df)
    vo_df["drift"] = drifts

    print("[DATASET] Loading entropy & uncertainty maps...")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)
    U_map = load_uncertainty_grid(coverage_csv)

    print("[DATASET] Extracting features...")
    entropies = []
    uncertainties = []
    speeds = []

    for i in range(len(vo_df)):
        x = vo_df.loc[i, "x"]
        y = vo_df.loc[i, "y"]

        entropies.append(extract_local_entropy(H_map, x, y))
        uncertainties.append(extract_local_uncertainty(U_map, x, y))

        if i == 0:
            speeds.append(0.0)
        else:
            dx = vo_df.loc[i, "x"] - vo_df.loc[i - 1, "x"]
            dy = vo_df.loc[i, "y"] - vo_df.loc[i - 1, "y"]
            speeds.append(math.sqrt(dx * dx + dy * dy))

    vo_df["entropy"] = entropies
    vo_df["local_uncertainty"] = uncertainties
    vo_df["speed"] = speeds

    # dataset columns
    dataset = vo_df[["entropy", "local_uncertainty", "speed", "drift"]]
    dataset.to_csv(output_csv, index=False)

    print(f"[DATASET] Saved drift dataset to: {output_csv}")
