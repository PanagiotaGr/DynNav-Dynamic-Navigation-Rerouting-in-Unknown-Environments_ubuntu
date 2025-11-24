#!/usr/bin/env python3
"""
evaluate_uapcr_metrics.py

Σκοπός:
- Να υπολογίσει βασικές μετρικές για την προτεινόμενη UAPCR μέθοδο:
  * ποσοστό κάλυψης πριν/μετά
  * μήκος διαδρομής (VO + weighted replan)
  * Normalized Coverage Efficiency (NCE)

Υποθέσεις:
- coverage_grid.csv περιέχει:
    ['cell_id', 'row', 'col', 'center_x', 'center_z', 'covered']
  όπου 'covered' είναι 0 ή 1 (ή γενικά [0,1]).
- high_priority_cells.csv περιέχει τα κελιά στα οποία γίνεται weighted replan.
  Μετά το merge έχουν στήλες τύπου: cell_id_cov, ..., priority.
- replan_weighted_waypoints.csv περιέχει τα weighted waypoints με στήλες ['x', 'z'].

Απλοποίηση:
- Θεωρούμε ότι όλα τα high-priority κελιά θα καλυφθούν επιτυχώς μετά το replan.
- Για VO path length χρησιμοποιούμε μια σταθερή τιμή (π.χ. όπως από άλλο script/log).
"""

import os
import math
import pandas as pd

COVERAGE_CSV = "coverage_grid.csv"
HIGH_PRIORITY_CSV = "high_priority_cells.csv"
WEIGHTED_REPLAN_CSV = "replan_weighted_waypoints.csv"

# Βάλε εδώ το μήκος της αρχικής τροχιάς VO (π.χ. 189.57 από το δικό σου log)
VO_PATH_LENGTH = 189.57  # τροποποίησέ το αν έχεις άλλο νούμερο


def compute_path_length_from_csv(csv_path, x_col="x", z_col="z"):
    """Υπολογίζει μήκος διαδρομής από CSV με στήλες x_col, z_col."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return 0.0

    xs = df[x_col].tolist()
    zs = df[z_col].tolist()

    total = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dz = zs[i] - zs[i - 1]
        total += math.sqrt(dx * dx + dz * dz)
    return total


def main():
    if not os.path.exists(COVERAGE_CSV):
        raise FileNotFoundError(f"Could not find {COVERAGE_CSV}")
    if not os.path.exists(HIGH_PRIORITY_CSV):
        raise FileNotFoundError(f"Could not find {HIGH_PRIORITY_CSV}")
    if not os.path.exists(WEIGHTED_REPLAN_CSV):
        raise FileNotFoundError(f"Could not find {WEIGHTED_REPLAN_CSV}")

    print(f"Loading coverage grid from {COVERAGE_CSV}...")
    cov_df = pd.read_csv(COVERAGE_CSV)
    print(f"Loaded {len(cov_df)} cells.")

    # Αρχική κάλυψη: ποσοστό κελιών με covered ∈ [0,1]
    if "covered" not in cov_df.columns:
        raise ValueError(f"'covered' column not found in {COVERAGE_CSV}. Columns: {cov_df.columns}")

    cov_df["covered"] = cov_df["covered"].clip(0.0, 1.0)
    coverage_before = cov_df["covered"].mean() * 100.0

    print(f"Coverage BEFORE weighted replan: {coverage_before:.2f}%")

    # Διαβάζουμε τα high-priority cells – υποθέτουμε ότι μετά το weighted replan καλύπτονται.
    print(f"Loading high-priority cells from {HIGH_PRIORITY_CSV}...")
    hp_df = pd.read_csv(HIGH_PRIORITY_CSV)
    print(f"[INFO] high_priority_cells.csv columns: {list(hp_df.columns)}")

    # Θα χρησιμοποιήσουμε το cell_id από την coverage_grid,
    # και το cell_id_cov από το high_priority_cells (όπως προέκυψε από το merge).
    cov_df_after = cov_df.copy()

    if "cell_id" in cov_df_after.columns and "cell_id_cov" in hp_df.columns:
        cov_df_after.set_index("cell_id", inplace=True)
        ids = hp_df["cell_id_cov"].unique().tolist()
        cov_df_after.loc[ids, "covered"] = 1.0
        coverage_after = cov_df_after["covered"].mean() * 100.0
    else:
        # Fallback: index-based, αν κάτι δεν πάει όπως περιμένουμε
        print("[WARN] 'cell_id' or 'cell_id_cov' not found. Falling back to index-based approximation.")
        hp_indices = hp_df.index
        cov_df_after = cov_df.copy()
        hp_indices = hp_indices.intersection(cov_df_after.index)
        cov_df_after.loc[hp_indices, "covered"] = 1.0
        coverage_after = cov_df_after["covered"].mean() * 100.0

    improvement = coverage_after - coverage_before

    print(f"Coverage AFTER weighted replan:  {coverage_after:.2f}%")
    print(f"Absolute improvement:            {improvement:.2f} percentage points")

    # Μήκος διαδρομής weighted replan
    weighted_path_length = compute_path_length_from_csv(WEIGHTED_REPLAN_CSV, x_col="x", z_col="z")
    total_path_length = VO_PATH_LENGTH + weighted_path_length

    print()
    print("=== Path length statistics ===")
    print(f"VO path length:          {VO_PATH_LENGTH:.2f}")
    print(f"Weighted replan length:  {weighted_path_length:.2f}")
    print(f"Total path length:       {total_path_length:.2f}")

    # Normalized Coverage Efficiency (NCE = coverage / path length)
    coverage_before_norm = coverage_before / 100.0
    coverage_after_norm = coverage_after / 100.0

    nce_before = coverage_before_norm / VO_PATH_LENGTH if VO_PATH_LENGTH > 0 else 0.0
    nce_after = coverage_after_norm / total_path_length if total_path_length > 0 else 0.0

    print()
    print("=== Normalized Coverage Efficiency (NCE) ===")
    print(f"NCE BEFORE: {nce_before:.6f}")
    print(f"NCE AFTER:  {nce_after:.6f}")

    print()
    print("Done evaluating UAPCR metrics.")


if __name__ == "__main__":
    main()
