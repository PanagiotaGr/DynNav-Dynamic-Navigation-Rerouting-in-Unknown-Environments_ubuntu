import csv
import math
import os
from datetime import datetime

import numpy as np


def load_coverage_grid(path="coverage_grid.csv"):
    rows = []
    covered = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["row"]), int(row["col"])))
            covered.append(bool(int(row["covered"])))
    covered = np.array(covered, dtype=bool)
    return rows, covered


def load_path_length_from_csv(path, has_header=True):
    """
    Διαβάζει waypoints από csv (id,x,z) και επιστρέφει συνολικό μήκος διαδρομής.
    Αν το αρχείο δεν υπάρχει → επιστρέφει 0.0.
    """
    if not os.path.exists(path):
        return 0.0

    xs = []
    zs = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            # row = [id, x, z]
            if len(row) < 3:
                continue
            xs.append(float(row[1]))
            zs.append(float(row[2]))
    if len(xs) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(xs)):
        total += math.dist((xs[i - 1], zs[i - 1]), (xs[i], zs[i]))
    return total


def load_weighted_eval(path="weighted_eval_report.csv"):
    """
    Φορτώνει τα metrics από evaluate_weighted_coverage.py
    Επιστρέφει dict με before, after, gain, path_length, efficiency.
    Αν το αρχείο λείπει → επιστρέφει None.
    """
    if not os.path.exists(path):
        return None

    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["metric"]
            val = float(row["value"])
            data[key] = val
    return data


def main():
    # 1. Coverage grid
    cov_cells, covered = load_coverage_grid("coverage_grid.csv")
    total_cells = len(cov_cells)
    base_coverage = covered.mean() if total_cells > 0 else 0.0

    # 2. Full replan path length (replan_missing_cells)
    full_replan_len = load_path_length_from_csv("replan_waypoints.csv", has_header=True)

    # 3. Weighted replan path length
    weighted_replan_len = load_path_length_from_csv("replan_weighted_waypoints.csv", has_header=True)

    # 4. Weighted eval metrics (high-priority coverage)
    w_eval = load_weighted_eval("weighted_eval_report.csv")

    hp_cells = None
    hp_before = None
    hp_after = None
    hp_gain = None
    hp_eff = None
    hp_path = None

    if w_eval is not None:
        hp_cells = w_eval.get("hp_cells", None)
        hp_before = w_eval.get("before", None)
        hp_after = w_eval.get("after", None)
        hp_gain = w_eval.get("absolute_gain", None)
        hp_eff = w_eval.get("efficiency", None)
        hp_path = w_eval.get("path_length", None)

    # 5. timestamp + simple id
    ts = datetime.now().isoformat(timespec="seconds")

    # 6. γράφουμε/προσθέτουμε στο experiments_log.csv
    log_path = "experiments_log.csv"
    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "total_cells",
                "base_coverage",
                "full_replan_path_length",
                "weighted_replan_path_length",
                "hp_cells",
                "hp_coverage_before",
                "hp_coverage_after",
                "hp_gain",
                "hp_path_length",
                "hp_efficiency",
            ])

        writer.writerow([
            ts,
            total_cells,
            base_coverage,
            full_replan_len,
            weighted_replan_len,
            hp_cells,
            hp_before,
            hp_after,
            hp_gain,
            hp_path,
            hp_eff,
        ])

    print("=== Experiment logged ===")
    print(f"Timestamp: {ts}")
    print(f"Total cells: {total_cells}")
    print(f"Base coverage: {100*base_coverage:.2f}%")
    print(f"Full replan path length: {full_replan_len:.2f}")
    print(f"Weighted replan path length: {weighted_replan_len:.2f}")
    if hp_cells is not None:
        print(f"High-priority cells: {int(hp_cells)}")
        print(f"HP coverage before: {100*hp_before:.2f}%")
        print(f"HP coverage after:  {100*hp_after:.2f}%")
        print(f"HP gain:            {100*hp_gain:.2f} % points")
        print(f"HP efficiency:      {hp_eff:.6f} coverage per meter")
    print(f"[OK] Appended row to {log_path}")


if __name__ == "__main__":
    main()
