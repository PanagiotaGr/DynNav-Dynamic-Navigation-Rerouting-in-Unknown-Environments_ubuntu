import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from info_gain_planner import (
    load_coverage_grid,
    load_uncertainty_grid,
    compute_entropy_map,
)


def find_frontiers(covered_grid: np.ndarray) -> np.ndarray:
    """
    Εντοπίζει frontier cells:
      frontier = κελί που είναι "unknown" αλλά έχει τουλάχιστον έναν γείτονα "covered".
    covered_grid: 2D array με 1 για covered, 0 για uncovered/unknown
    Επιστρέφει 2D boolean mask ίδιου σχήματος.
    """
    rows, cols = covered_grid.shape
    frontier = np.zeros_like(covered_grid, dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if covered_grid[r, c] > 0.5:
                continue  # already covered

            # unknown cell → check neighbors
            has_covered_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    if covered_grid[rr, cc] > 0.5:
                        has_covered_neighbor = True
                        break

            if has_covered_neighbor:
                frontier[r, c] = True

    return frontier


def connected_components(mask: np.ndarray) -> list:
    """
    BFS για connected components σε boolean mask.
    Επιστρέφει λίστα από components, όπου κάθε component είναι
    λίστα από (row, col) tuples.
    """
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue

            queue = [(r, c)]
            visited[r, c] = True
            comp = []

            while queue:
                cr, cc = queue.pop(0)
                comp.append((cr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = cr + dr
                    cc = cc = cc  # dummy
            # Oops, να το δώσω σωστό από την αρχή χωρίς μπέρδεμα:


def connected_components(mask: np.ndarray) -> list:
    """
    BFS για connected components σε boolean mask.
    Επιστρέφει λίστα από components, όπου κάθε component είναι
    λίστα από (row, col) tuples.
    """
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue

            queue = [(r, c)]
            visited[r, c] = True
            comp = []

            while queue:
                cr, cc = queue.pop(0)
                comp.append((cr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = cr + dr
                    cc = cc + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if mask[rr, cc] and not visited[rr, cc]:
                            visited[rr, cc] = True
                            queue.append((rr, cc))

            components.append(comp)

    return components


def frontier_clusters_stats(components, H_map, U_map, robot_row=0, robot_col=0):
    """
    Για κάθε frontier cluster υπολογίζει:
      - centroid (row,col)
      - mean entropy
      - mean uncertainty
      - distance από ρομπότ
    Επιστρέφει λίστα dictionaries.
    """
    stats = []
    for comp in components:
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))

        H_vals = [H_map[r, c] for r, c in comp]
        U_vals = [U_map[r, c] for r, c in comp]

        mean_H = float(np.mean(H_vals))
        mean_U = float(np.mean(U_vals))

        dist = math.sqrt((centroid_row - robot_row) ** 2 +
                         (centroid_col - robot_col) ** 2)

        stats.append({
            "centroid_row": centroid_row,
            "centroid_col": centroid_col,
            "mean_entropy": mean_H,
            "mean_uncertainty": mean_U,
            "distance": dist,
        })

    return stats


def score_frontier_cluster(fc, w_H=1.0, w_U=1.0, w_cost=0.2, eps=1e-6):
    """
    Multi-objective score για frontier cluster.
    """
    benefit = w_H * fc["mean_entropy"] + w_U * fc["mean_uncertainty"]
    cost = w_cost * fc["distance"]
    return benefit / (cost + eps)


if __name__ == "__main__":
    coverage_csv = "coverage_grid_with_uncertainty_vla.csv"
    output_csv = "frontier_nbv_goal.csv"
    output_fig = "frontier_nbv_viz.png"

    print("[FRONTIER] Loading coverage with uncertainty...")
    prob_grid = load_coverage_grid(coverage_csv)
    H_map = compute_entropy_map(prob_grid)
    U_map = load_uncertainty_grid(coverage_csv)

    # covered / unknown από το probability grid:
    # εδώ: p≈0.05 -> covered, p≈0.5 -> unknown
    covered_grid = (prob_grid < 0.3).astype(float)

    # 1) frontier cells
    frontier_mask = find_frontiers(covered_grid)
    print(f"[FRONTIER] Frontier cells: {np.sum(frontier_mask)}")

    # 2) connected components / clusters
    components = connected_components(frontier_mask)
    print(f"[FRONTIER] Frontier clusters: {len(components)}")

    if not components:
        print("[FRONTIER] WARNING: no frontier clusters found.")
        exit(0)

    # 3) stats ανά cluster
    stats = frontier_clusters_stats(components, H_map, U_map,
                                    robot_row=0, robot_col=0)

    # 4) scoring
    w_H, w_U, w_cost = 1.0, 1.0, 0.2
    best_score = -1e9
    best_fc = None

    for fc in stats:
        fc["score"] = score_frontier_cluster(fc, w_H, w_U, w_cost)
        if fc["score"] > best_score:
            best_score = fc["score"]
            best_fc = fc

    print("[FRONTIER] Best frontier cluster:")
    print(best_fc)

    # 5) Save goal
    goal_df = pd.DataFrame([{
        "goal_row": best_fc["centroid_row"],
        "goal_col": best_fc["centroid_col"],
        "goal_x": best_fc["centroid_col"],
        "goal_y": best_fc["centroid_row"],
        "score": best_fc["score"]
    }])
    goal_df.to_csv(output_csv, index=False)
    print(f"[FRONTIER] Saved goal to: {output_csv}")

    # 6) Visualization
    rows, cols = H_map.shape
    plt.figure(figsize=(7, 7))
    plt.imshow(H_map, origin="lower", cmap="viridis")
    plt.colorbar(label="Entropy")

    fr_rows, fr_cols = np.where(frontier_mask)
    plt.scatter(fr_cols, fr_rows, s=10, c="white", alpha=0.5, label="Frontier cells")

    plt.scatter(best_fc["centroid_col"], best_fc["centroid_row"],
                s=160, marker="*", c="red", label="Best frontier goal")

    plt.title("Uncertainty-aware Frontier Exploration")
    plt.xlabel("X (grid col)")
    plt.ylabel("Y (grid row)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_fig, dpi=220)
    plt.show()
    print(f"[FRONTIER] Saved visualization to: {output_fig}")
