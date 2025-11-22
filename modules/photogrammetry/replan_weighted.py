import csv
import math
from typing import List, Tuple


# threshold προτεραιότητας (σε [0,1], επειδή έχουμε normalized P)
MIN_PRIORITY = 0.4
UAV_SPEED = 5.0  # "m/s" ή VO units per second


def load_priority_field(csv_path="priority_field.csv"):
    """
    Φορτώνει το priority_field.csv και επιστρέφει λίστα:
    (row, col, center_x, center_z, priority)
    """
    cells = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["row"])
            c = int(row["col"])
            cx = float(row["center_x"])
            cz = float(row["center_z"])
            p = float(row["priority"])
            cells.append((r, c, cx, cz, p))
    return cells


def filter_high_priority_cells(cells, min_priority=MIN_PRIORITY):
    """
    Κρατά μόνο τα κελιά με priority >= min_priority.
    """
    selected = [
        (r, c, cx, cz, p)
        for (r, c, cx, cz, p) in cells
        if p >= min_priority
    ]
    return selected


def boustrophedon_path(cells):
    """
    Δημιουργεί ζιγκ-ζαγκ (boustrophedon) διαδρομή πάνω στα επιλεγμένα κελιά.

    Επιστρέφει λίστα από (x, z) waypoints.
    """
    if not cells:
        return []

    # group ανά row
    rows = {}
    for (r, c, cx, cz, p) in cells:
        rows.setdefault(r, []).append((c, cx, cz, p))

    # sort rows από μικρό προς μεγάλο index (όπως grid)
    sorted_rows = sorted(rows.items(), key=lambda x: x[0])  # (row_idx, [(c, cx, cz, p)...])

    waypoints = []
    for i, (r, cols_list) in enumerate(sorted_rows):
        # sort columns από αριστερά προς δεξιά
        cols_sorted = sorted(cols_list, key=lambda x: x[0])

        # Για boustrophedon: κάθε δεύτερη γραμμή ανάποδα
        if i % 2 == 1:
            cols_sorted = list(reversed(cols_sorted))

        for (c, cx, cz, p) in cols_sorted:
            waypoints.append((cx, cz))

    return waypoints


def path_length(waypoints: List[Tuple[float, float]]) -> float:
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(waypoints)):
        x1, z1 = waypoints[i - 1]
        x2, z2 = waypoints[i]
        total += math.dist((x1, z1), (x2, z2))
    return total


def save_waypoints(waypoints, csv_path="replan_weighted_waypoints.csv"):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "z"])
        for i, (x, z) in enumerate(waypoints):
            writer.writerow([i, x, z])
    print(f"[OK] Saved {len(waypoints)} weighted replan waypoints to {csv_path}")


def main():
    # 1. Φόρτωση priority field
    cells = load_priority_field("priority_field.csv")
    print(f"Loaded {len(cells)} cells from priority_field.csv")

    # 2. Επιλογή high-priority cells
    selected = filter_high_priority_cells(cells, MIN_PRIORITY)
    print(f"Selected {len(selected)} cells with priority >= {MIN_PRIORITY}")

    if not selected:
        print("No high-priority cells found. Nothing to replan.")
        return

    # 3. Δημιουργία boustrophedon διαδρομής
    wps = boustrophedon_path(selected)
    print(f"Generated {len(wps)} weighted replan waypoints")

    # 4. Υπολογισμός μήκους & χρόνου
    L = path_length(wps)
    t = L / UAV_SPEED if UAV_SPEED > 0 else 0.0

    print("=== Weighted Replan Path Statistics ===")
    print(f"Total path length: {L:.2f} (VO units)")
    print(f"Estimated time:   {t:.1f} s  (~{t/60:.1f} min)")

    # 5. Αποθήκευση
    save_waypoints(wps, "replan_weighted_waypoints.csv")


if __name__ == "__main__":
    main()
