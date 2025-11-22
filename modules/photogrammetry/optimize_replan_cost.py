import csv
import math
from typing import List, Tuple

import numpy as np


# ====== ΠΑΡΑΜΕΤΡΟΙ ΚΟΣΤΟΥΣ (μπορείς να τις αλλάξεις / κάνεις πειράματα) ======
ALPHA = 1.0    # βάρος για coverage gain
BETA = 0.001   # βάρος για path length (τιμωρεί μεγάλες διαδρομές)
GAMMA = 0.5    # βάρος για residual uncertainty

VISIT_RADIUS = 2.0   # απόσταση για να θεωρήσουμε ότι ένα κελί "επισκέφθηκε"
UAV_SPEED = 5.0      # για πληροφοριακούς χρόνους, όχι απαραίτητο για J

# thresholds προτεραιότητας που θα δοκιμάσουμε
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]


def load_priority_field(csv_path="priority_field.csv"):
    """
    Επιστρέφει:
      cells: [(row, col)]
      P:    priority per cell (np.array)
      cx, cz: center coords per cell (np.array)
    """
    rows = []
    cols = []
    P = []
    cx = []
    cz = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["row"])
            c = int(row["col"])
            rows.append(r)
            cols.append(c)
            cx.append(float(row["center_x"]))
            cz.append(float(row["center_z"]))
            P.append(float(row["priority"]))

    cells = list(zip(rows, cols))
    return cells, np.array(P), np.array(cx), np.array(cz)


def load_uncertainty_map(csv_path="feature_uncertainty_grid.csv"):
    """
    Επιστρέφει dict: (row,col) -> uncertainty
    """
    u_map = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["row"])
            c = int(row["col"])
            u = float(row["uncertainty"])
            u_map[(r, c)] = u
    return u_map


def boustrophedon_path(selected_cells):
    """
    selected_cells: [(row, col, cx, cz, Pvalue)]
    Επιστρέφει λίστα (x,z) waypoints σε ζιγκ-ζαγκ.
    """
    if not selected_cells:
        return []

    rows = {}
    for (r, c, cx, cz, p) in selected_cells:
        rows.setdefault(r, []).append((c, cx, cz, p))

    sorted_rows = sorted(rows.items(), key=lambda x: x[0])  # (row_idx, list)

    waypoints = []
    for i, (r, cols_list) in enumerate(sorted_rows):
        cols_sorted = sorted(cols_list, key=lambda x: x[0])
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


def evaluate_for_threshold(
    cells, P, cx, cz, u_map, threshold: float
):
    """
    Για δοσμένο threshold:
      - επιλέγει κελιά με P >= threshold
      - παράγει boustrophedon path
      - μετρά coverage over αυτά τα κελιά
      - residual uncertainty
      - path length
      - cost J
    """

    # 1. Επιλογή high-priority cells
    mask = P >= threshold
    hp_indices = np.where(mask)[0]

    if len(hp_indices) == 0:
        return {
            "threshold": threshold,
            "hp_cells": 0,
            "coverage_gain": 0.0,
            "residual_uncertainty": 0.0,
            "path_length": 0.0,
            "J": -np.inf,
        }

    selected = []
    hp_uncertainties = []
    for idx in hp_indices:
        r, c = cells[idx]
        x = cx[idx]
        z = cz[idx]
        p_val = P[idx]
        u_val = u_map.get((r, c), 1.0)  # default uncertainty 1 αν δεν βρεθεί
        selected.append((r, c, x, z, p_val))
        hp_uncertainties.append(u_val)

    hp_uncertainties = np.array(hp_uncertainties)

    # 2. Παράγουμε boustrophedon path
    wps = boustrophedon_path(selected)
    if not wps:
        return {
            "threshold": threshold,
            "hp_cells": len(hp_indices),
            "coverage_gain": 0.0,
            "residual_uncertainty": float(hp_uncertainties.mean()),
            "path_length": 0.0,
            "J": -np.inf,
        }

    wp_x = np.array([p[0] for p in wps])
    wp_z = np.array([p[1] for p in wps])

    # 3. coverage των HP κελιών ως "επισκέφθηκαν ή όχι"
    visited_mask = []
    for idx in hp_indices:
        x = cx[idx]
        z = cz[idx]
        d2 = (wp_x - x) ** 2 + (wp_z - z) ** 2
        visited = d2.min() <= VISIT_RADIUS**2
        visited_mask.append(visited)
    visited_mask = np.array(visited_mask, dtype=bool)

    coverage_gain = visited_mask.mean()  # 0..1 (πόσα HP καλύψαμε)
    # residual uncertainty = mean U στα μη επισκεφθέντα high-priority cells
    if (~visited_mask).any():
        residual_uncertainty = float(hp_uncertainties[~visited_mask].mean())
    else:
        residual_uncertainty = 0.0

    # 4. path length
    L = path_length(wps)

    # 5. cost J
    J = ALPHA * coverage_gain - BETA * L - GAMMA * residual_uncertainty

    return {
        "threshold": threshold,
        "hp_cells": len(hp_indices),
        "coverage_gain": coverage_gain,
        "residual_uncertainty": residual_uncertainty,
        "path_length": L,
        "J": J,
    }


def main():
    # Φόρτωση δεδομένων
    cells, P, cx, cz = load_priority_field("priority_field.csv")
    u_map = load_uncertainty_map("feature_uncertainty_grid.csv")

    results = []

    print("=== Multi-objective optimization over priority thresholds ===")
    print(f"(ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA})")
    print("Threshold | HP cells | CovGain | ResidU | PathLen | J")

    for thr in THRESHOLDS:
        res = evaluate_for_threshold(cells, P, cx, cz, u_map, thr)
        results.append(res)
        print(
            f"{res['threshold']:.2f}      | "
            f"{res['hp_cells']:7d}  | "
            f"{res['coverage_gain']:.3f}  | "
            f"{res['residual_uncertainty']:.3f} | "
            f"{res['path_length']:.1f} | "
            f"{res['J']:.4f}"
        )

    # βρίσκουμε το καλύτερο J
    best = max(results, key=lambda r: r["J"])
    print("\n=== BEST CONFIGURATION ===")
    print(
        f"Best threshold: {best['threshold']:.2f}\n"
        f"High-priority cells: {best['hp_cells']}\n"
        f"Coverage gain: {best['coverage_gain']*100:.2f}%\n"
        f"Residual uncertainty: {best['residual_uncertainty']:.3f}\n"
        f"Path length: {best['path_length']:.2f}\n"
        f"Cost J: {best['J']:.4f}"
    )

    # αποθήκευση αποτελεσμάτων σε CSV για plotting
    with open("multiobj_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "hp_cells",
                "coverage_gain",
                "residual_uncertainty",
                "path_length",
                "J",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["threshold"],
                    r["hp_cells"],
                    r["coverage_gain"],
                    r["residual_uncertainty"],
                    r["path_length"],
                    r["J"],
                ]
            )

    print("[OK] Saved multiobj_results.csv")


if __name__ == "__main__":
    main()
