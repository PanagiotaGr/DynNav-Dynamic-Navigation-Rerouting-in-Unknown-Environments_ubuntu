import csv
import math
from typing import List, Tuple


def load_coverage_grid(csv_path: str = "coverage_grid.csv"):
    """
    Φορτώνει το coverage_grid.csv και επιστρέφει λίστα από cells:
    (cell_id, row, col, center_x, center_z, covered)
    """
    cells = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = int(row["cell_id"])
            r = int(row["row"])
            c = int(row["col"])
            cx = float(row["center_x"])
            cz = float(row["center_z"])
            covered = bool(int(row["covered"]))
            cells.append((cell_id, r, c, cx, cz, covered))
    return cells


def filter_uncovered_cells(cells, min_margin_rows=0, min_margin_cols=0):
    """
    Κρατά μόνο τα cells με covered=False.
    Προαιρετικά μπορούμε να αγνοήσουμε κάποια border rows/cols (min_margin_*),
    αν δεν μας ενδιαφέρουν.
    """
    if not cells:
        return []

    rows = [c[1] for c in cells]
    cols = [c[2] for c in cells]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    r_lo = min_row + min_margin_rows
    r_hi = max_row - min_margin_rows
    c_lo = min_col + min_margin_cols
    c_hi = max_col - min_margin_cols

    uncovered = []
    for cell_id, r, c, cx, cz, covered in cells:
        if covered:
            continue
        if r < r_lo or r > r_hi:
            continue
        if c < c_lo or c > c_hi:
            continue
        uncovered.append((cell_id, r, c, cx, cz))

    return uncovered


def boustrophedon_order(uncovered_cells):
    """
    Δημιουργεί μια "ζιγκ-ζαγκ" (boustrophedon) σειρά
    πάνω από τα ανεπίσκεπτα cells.

    Επιστρέφει λίστα από (cx, cz) ως waypoints.
    """

    if not uncovered_cells:
        return []

    # group ανά row
    rows = {}
    for (cell_id, r, c, cx, cz) in uncovered_cells:
        rows.setdefault(r, []).append((c, cx, cz))

    # ταξινόμηση rows
    sorted_rows = sorted(rows.items(), key=lambda x: x[0])  # (row_index, [(c, cx, cz)...])

    waypoints = []
    for idx, (r, cols_list) in enumerate(sorted_rows):
        # sort columns
        cols_sorted = sorted(cols_list, key=lambda x: x[0])

        if idx % 2 == 1:
            # reverse for boustrophedon
            cols_sorted = list(reversed(cols_sorted))

        for (c, cx, cz) in cols_sorted:
            waypoints.append((cx, cz))

    return waypoints


def estimate_path_length(waypoints: List[Tuple[float, float]]):
    """
    Υπολογίζει το συνολικό μήκος της διαδρομής σε ίδιες μονάδες με τα coords.
    """
    if len(waypoints) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(waypoints)):
        x1, z1 = waypoints[i - 1]
        x2, z2 = waypoints[i]
        total += math.dist((x1, z1), (x2, z2))

    return total


def save_replan_waypoints(waypoints, csv_path="replan_waypoints.csv"):
    """
    Σώζει τα replan waypoints σε CSV: id, x, z.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "z"])
        for i, (x, z) in enumerate(waypoints):
            writer.writerow([i, x, z])

    print(f"Saved {len(waypoints)} replan waypoints to {csv_path}")


def main():
    # 1. Φόρτωση coverage grid
    cells = load_coverage_grid("coverage_grid.csv")
    print(f"Loaded {len(cells)} cells from coverage_grid.csv")

    # 2. Επιλογή uncovered cells (μπορούμε να παίξουμε με τα margins)
    uncovered = filter_uncovered_cells(cells, min_margin_rows=0, min_margin_cols=0)
    print(f"Uncovered cells (after margins): {len(uncovered)}")

    if not uncovered:
        print("No uncovered cells found. No replanning needed.")
        return

    # 3. Δημιουργία boustrophedon διαδρομής πάνω στα uncovered cells
    replan_wps = boustrophedon_order(uncovered)
    print(f"Generated {len(replan_wps)} replan waypoints")

    # 4. Εκτίμηση μήκους και χρόνου διαδρομής
    total_length = estimate_path_length(replan_wps)
    speed_m_s = 5.0  # υποθέτουμε 5 m/s (ή αντίστοιχη μονάδα VO)
    time_s = total_length / speed_m_s if speed_m_s > 0 else 0.0

    print("=== Replan path statistics ===")
    print(f"Total path length: {total_length:.2f} (VO units)")
    print(f"Estimated time:   {time_s:.1f} s  (~{time_s/60:.1f} min)")

    # 5. Αποθήκευση replan waypoints
    save_replan_waypoints(replan_wps, "replan_waypoints.csv")


if __name__ == "__main__":
    main()
