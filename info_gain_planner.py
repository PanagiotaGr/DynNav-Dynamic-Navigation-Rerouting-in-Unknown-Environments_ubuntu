import numpy as np
import pandas as pd
import math

# ============================================================
# 1. Entropy per cell (binary entropy, σε bits)
#    H(p) = - p log2 p - (1-p) log2 (1-p)
#    όπου p ~ "πιθανότητα άγνοιας / αβεβαιότητας" του κελιού
# ============================================================
def cell_entropy(p: float) -> float:
    # Αποφυγή log(0)
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return - (p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ============================================================
# 2. Φόρτωση coverage grid και κατασκευή prob_grid
#    prob_grid[row, col] = p_unknown(i,j) ∈ [0, 1]
#
#    Αν υπάρχει στήλη "uncertainty":
#      p_unknown = α * (1 - covered) + β * uncertainty
#    Αλλιώς:
#      p_unknown = 1 - covered  (0 = πλήρως καλυμμένο, 1 = ακάλυπτο)
# ============================================================
def load_coverage_grid(csv_path: str,
                       alpha: float = 0.7,
                       beta: float = 0.3) -> np.ndarray:
    """
    Φορτώνει coverage_grid(_with_uncertainty).csv και κατασκευάζει grid
    με πιθανότητα άγνοιας p_unknown για κάθε κελί.
    - covered ∈ {0,1} ή [0,1]
    - uncertainty ∈ [0,1] (αν υπάρχει)
    """
    df = pd.read_csv(csv_path)
    required_cols = ['row', 'col', 'covered']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {csv_path}")

    max_row = int(df['row'].max())
    max_col = int(df['col'].max())

    prob_grid = np.zeros((max_row + 1, max_col + 1), dtype=float)

    has_uncertainty = 'uncertainty' in df.columns
    if has_uncertainty:
        print("[INFO] Using 'uncertainty' column from coverage grid.")
    else:
        print("[INFO] No 'uncertainty' column, using only covered/uncovered.")

    for _, r in df.iterrows():
        row = int(r['row'])
        col = int(r['col'])
        covered = float(r['covered'])

        # clamp covered in [0,1]
        covered = max(0.0, min(1.0, covered))

        if has_uncertainty:
            u = float(r['uncertainty'])
            u = max(0.0, min(1.0, u))

            # Συνδυασμός μη-κάλυψης και VO-uncertainty
            # p_unknown = α * (1 - covered) + β * u
            p_unknown = alpha * (1.0 - covered) + beta * u
        else:
            # Fallback: μόνο από την κάλυψη
            # (0 = πλήρως καλυμμένο, 1 = ακάλυπτο)
            p_unknown = 1.0 - covered

        # clamp σε [0.01, 0.99] για πιο “υγιή” entropy
        p_unknown = max(0.01, min(0.99, p_unknown))

        prob_grid[row, col] = p_unknown

    return prob_grid


# ============================================================
# 3. Φόρτωση VO-based uncertainty map U(i,j)
# ============================================================
def load_uncertainty_grid(csv_path: str) -> np.ndarray:
    """
    Φορτώνει grid αβεβαιότητας U_map από coverage_grid(_with_uncertainty).csv.
    Αν δεν υπάρχει στήλη 'uncertainty', επιστρέφει μηδενικό grid.
    """
    df = pd.read_csv(csv_path)
    required_cols = ['row', 'col']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {csv_path}")

    max_row = int(df['row'].max())
    max_col = int(df['col'].max())
    U_map = np.zeros((max_row + 1, max_col + 1), dtype=float)

    if 'uncertainty' not in df.columns:
        print("[INFO] load_uncertainty_grid: no 'uncertainty' column, returning zeros.")
        return U_map

    print("[INFO] load_uncertainty_grid: using 'uncertainty' column.")
    for _, r in df.iterrows():
        row = int(r['row'])
        col = int(r['col'])
        u = float(r['uncertainty'])
        u = max(0.0, min(1.0, u))
        U_map[row, col] = u

    return U_map


# ============================================================
# 4. Υπολογισμός entropy map H(i,j) από prob_grid
# ============================================================
def compute_entropy_map(prob_grid: np.ndarray) -> np.ndarray:
    H = np.zeros_like(prob_grid)
    rows, cols = prob_grid.shape
    for i in range(rows):
        for j in range(cols):
            p = prob_grid[i, j]
            H[i, j] = cell_entropy(p)
    return H


# ============================================================
# 5. Priority map: fusion entropy + VO uncertainty
#    P(i,j) = w_H * H(i,j) + w_U * U(i,j)
# ============================================================
def compute_priority_map(entropy_map: np.ndarray,
                         uncertainty_map: np.ndarray,
                         w_entropy: float = 1.0,
                         w_uncertainty: float = 1.0) -> np.ndarray:
    """
    Υπολογίζει priority field:
      priority = w_entropy * H + w_uncertainty * U
    όπου H = entropy map, U = VO-based uncertainty map.
    """
    if entropy_map.shape != uncertainty_map.shape:
        raise ValueError("entropy_map and uncertainty_map must have same shape")
    return w_entropy * entropy_map + w_uncertainty * uncertainty_map


# ============================================================
# 6. Information gain γύρω από υποψήφιο στόχο
#    IG(c) = sum P(i,j) σε παράθυρο γύρω από το c
# ============================================================
def information_gain(priority_map: np.ndarray,
                     center_row: int,
                     center_col: int,
                     window_radius: int = 4) -> float:

    rows, cols = priority_map.shape
    r_min = max(0, center_row - window_radius)
    r_max = min(rows - 1, center_row + window_radius)
    c_min = max(0, center_col - window_radius)
    c_max = min(cols - 1, center_col + window_radius)

    patch = priority_map[r_min:r_max+1, c_min:c_max+1]
    return float(patch.sum())


# ============================================================
# 7. Φόρτωση cluster centroids από CSV
# ============================================================
def load_cluster_centroids(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ['x', 'y']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {csv_path}")
    return df


# ============================================================
# 8. Γεννήτρια candidate viewpoints γύρω από κάθε cluster
#    NBV-style: k σημεία σε δακτύλιο ακτίνας R γύρω από τον centroid
# ============================================================
def generate_candidate_viewpoints(centroids_df: pd.DataFrame,
                                  radius: float = 3.0,
                                  num_angles: int = 8) -> pd.DataFrame:
    """
    Παράγει πολλαπλά candidate viewpoints γύρω από κάθε cluster centroid.
    Επιστρέφει DataFrame με στήλες:
      cluster_x, cluster_y, vx, vy
    """
    candidates = []
    angles = np.linspace(0.0, 2.0 * math.pi, num_angles, endpoint=False)

    for _, r in centroids_df.iterrows():
        cx = float(r['x'])
        cy = float(r['y'])

        for theta in angles:
            vx = cx + radius * math.cos(theta)
            vy = cy + radius * math.sin(theta)
            candidates.append({
                'cluster_x': cx,
                'cluster_y': cy,
                'vx': vx,
                'vy': vy
            })

    return pd.DataFrame(candidates)


# ============================================================
# 9. Επιλογή επόμενου στόχου (NBV-style)
#    score(v) = IG(v) / (d(v) + ε)
# ============================================================
def pick_next_viewpoint(priority_map: np.ndarray,
                        candidates_df: pd.DataFrame,
                        robot_row: int,
                        robot_col: int,
                        window_radius: int = 4,
                        dist_epsilon: float = 1e-3):

    rows, cols = priority_map.shape

    best_score = -1.0
    best_info = None

    for _, r in candidates_df.iterrows():
        vx = float(r['vx'])
        vy = float(r['vy'])

        # world -> grid indices
        c_row = int(round(vy))
        c_col = int(round(vx))

        # αγνοούμε candidates εκτός grid
        if c_row < 0 or c_row >= rows or c_col < 0 or c_col >= cols:
            continue

        ig = information_gain(priority_map, c_row, c_col, window_radius)
        dist = math.sqrt((c_row - robot_row)**2 + (c_col - robot_col)**2)
        score = ig / (dist + dist_epsilon)

        if score > best_score:
            best_score = score
            best_info = {
                'goal_row': c_row,
                'goal_col': c_col,
                'goal_x': vx,
                'goal_y': vy,
                'information_gain': ig,
                'distance': dist,
                'score': score,
                'cluster_x': float(r['cluster_x']),
                'cluster_y': float(r['cluster_y'])
            }

    return best_info


# ============================================================
# 10. Demo main
# ============================================================
if __name__ == "__main__":
    coverage_path = "coverage_grid_with_uncertainty_pose.csv"
    centroids_csv = "cluster_waypoints.csv"
    output_csv = "ig_next_goal.csv"

    print(f"Loading coverage grid from: {coverage_csv}")
    prob_grid = load_coverage_grid(coverage_csv,
                                   alpha=0.7,  # βάρος μη-κάλυψης
                                   beta=0.3)   # βάρος VO-uncertainty

    H_map = compute_entropy_map(prob_grid)

    # Φόρτωση uncertainty grid
    print("Loading uncertainty grid (U_map)...")
    U_map = load_uncertainty_grid(coverage_csv)

    # ΣΥΝΔΥΑΣΜΟΣ: priority = w_H * entropy + w_U * uncertainty
    w_H = 1.0
    w_U = 1.0
    priority_map = compute_priority_map(H_map, U_map,
                                        w_entropy=w_H,
                                        w_uncertainty=w_U)
    print(f"Using priority map with weights: w_H={w_H}, w_U={w_U}")

    print(f"Loading centroids from: {centroids_csv}")
    centroids_df = load_cluster_centroids(centroids_csv)

    # Παραγωγή candidate viewpoints (NBV-style)
    candidates_df = generate_candidate_viewpoints(
        centroids_df,
        radius=3.0,     # σε grid units
        num_angles=8    # 8 σημεία γύρω από κάθε centroid
    )
    print(f"Generated {len(candidates_df)} candidate viewpoints.")

    # Παράδειγμα: θέση ρομπότ στο (0, 0) σε grid coordinates
    robot_row, robot_col = 0, 0

    result = pick_next_viewpoint(
        priority_map,
        candidates_df,
        robot_row,
        robot_col,
        window_radius=4
    )

    print("\n======================")
    print("Next viewpoint (NBV-style, entropy + uncertainty):")
    print(result)
    print("======================\n")

    if result is not None:
        goal_df = pd.DataFrame([result])
        goal_df.to_csv(output_csv, index=False)
        print(f"Saved next goal to: {output_csv}")
    else:
        print("[WARNING] No valid candidate viewpoints found.")
