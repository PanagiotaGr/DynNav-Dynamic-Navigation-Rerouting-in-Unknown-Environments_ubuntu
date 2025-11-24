#!/usr/bin/env python3
"""
replan_weighted_uapcr.py

Σκοπός:
- Δημιουργεί μια "weighted" διαδρομή επανασχεδιασμού (replan) πάνω στα κελιά
  υψηλής προτεραιότητας, όπως προκύπτουν από το priority field (UAPCR).
- Χρησιμοποιεί το αρχείο high_priority_cells.csv, το οποίο έχει παραχθεί από το
  compute_priority_field_uapcr.py.

Τεχνικές λεπτομέρειες:
- Επειδή το compute_priority_field_uapcr.py κάνει merge coverage + uncertainty,
  τα ονόματα των στηλών είναι τύπου:
    row_cov, col_cov, center_x_cov, center_z_cov
  και όχι απλά row, col, center_x, center_z.
- Το script προσπαθεί να βρει αυτόματα τα σωστά ονόματα (row_cov / row, κ.λπ.).
- Η διαδρομή είναι boustrophedon-style (lawnmower/snake) πάνω στις γραμμές (rows).

Έξοδος:
- replan_weighted_waypoints.csv: λίστα waypoints της weighted διαδρομής επανασχεδιασμού.
"""

import os
import math
import pandas as pd

HIGH_PRIORITY_CSV = "high_priority_cells.csv"
WEIGHTED_REPLAN_CSV = "replan_weighted_waypoints.csv"

# Υποθέτουμε σταθερή "γραμμική" ταχύτητα σε VO units/sec (για εκτίμηση χρόνου)
ROBOT_SPEED = 5.0  # VO units per second


def compute_path_length(xs, zs):
    """Υπολογίζει το συνολικό μήκος διαδρομής για λίστα από (x, z) σημεία."""
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dz = zs[i] - zs[i - 1]
        total += math.sqrt(dx * dx + dz * dz)
    return total


def main():
    if not os.path.exists(HIGH_PRIORITY_CSV):
        raise FileNotFoundError(f"Could not find {HIGH_PRIORITY_CSV}")

    print(f"Loading high-priority cells from {HIGH_PRIORITY_CSV}...")
    df = pd.read_csv(HIGH_PRIORITY_CSV)
    print(f"[INFO] Columns in high_priority_cells.csv: {list(df.columns)}")

    # Πιθανά ονόματα στηλών (λόγω merge με suffixes)
    possible_cols = {
        "row": ["row_cov", "row", "row_unc"],
        "col": ["col_cov", "col", "col_unc"],
        "x": ["center_x_cov", "center_x", "center_x_unc"],
        "z": ["center_z_cov", "center_z", "center_z_unc"],
    }

    # Εντοπισμός πραγματικών ονομάτων στηλών
    def pick_col(candidates, desc):
        for c in candidates:
            if c in df.columns:
                print(f"[INFO] Using '{c}' as {desc} column.")
                return c
        raise ValueError(
            f"Could not find a suitable {desc} column. "
            f"Tried {candidates}, found columns: {list(df.columns)}"
        )

    col_row = pick_col(possible_cols["row"], "row")
    col_col = pick_col(possible_cols["col"], "col")
    col_x = pick_col(possible_cols["x"], "x (center_x)")
    col_z = pick_col(possible_cols["z"], "z (center_z)")

    # Ταξινόμηση ανά row με boustrophedon pattern
    print("Building boustrophedon-style path over high-priority cells...")
    all_rows = sorted(df[col_row].unique())

    xs = []
    zs = []
    waypoint_rows = []
    waypoint_cols = []

    for r in all_rows:
        row_cells = df[df[col_row] == r].copy()
        if len(row_cells) == 0:
            continue

        # Άρτιες γραμμές: col αύξουσα, περιττές: col φθίνουσα
        if int(r) % 2 == 0:
            row_cells = row_cells.sort_values(by=col_col, ascending=True)
        else:
            row_cells = row_cells.sort_values(by=col_col, ascending=False)

        xs.extend(row_cells[col_x].tolist())
        zs.extend(row_cells[col_z].tolist())
        waypoint_rows.extend(row_cells[col_row].tolist())
        waypoint_cols.extend(row_cells[col_col].tolist())

    # Υπολογισμός συνολικού μήκους διαδρομής
    total_length = compute_path_length(xs, zs)
    est_time = total_length / ROBOT_SPEED if ROBOT_SPEED > 0 else 0.0

    print("=== Weighted Replan Path Statistics (UAPCR) ===")
    print(f"Number of waypoints: {len(xs)}")
    print(f"Total path length:   {total_length:.2f} (VO units)")
    print(f"Estimated time:      {est_time:.1f} s  (~{est_time/60.0:.1f} min)")
    print()

    # Αποθήκευση waypoints σε CSV
    out_df = pd.DataFrame({
        "waypoint_id": list(range(len(xs))),
        "row": waypoint_rows,
        "col": waypoint_cols,
        "x": xs,
        "z": zs,
    })

    print(f"Saving weighted replan waypoints to {WEIGHTED_REPLAN_CSV}...")
    out_df.to_csv(WEIGHTED_REPLAN_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
