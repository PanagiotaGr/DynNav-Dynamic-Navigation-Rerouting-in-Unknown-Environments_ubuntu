import numpy as np
from astar_learned_heuristic import make_grid, astar_classic


def get_free_cells(grid):
    """Λίστα (x, y) για όλα τα free κελιά (0)."""
    free = np.argwhere(grid == 0)  # (row, col) = (y, x)
    return [(int(c[1]), int(c[0])) for c in free]


def compute_features(x, y, gx, gy, grid):
    """
    Rich features 11D για node (x,y) προς goal (gx,gy).
    ΠΡΕΠΕΙ να είναι ίδια με αυτά που χρησιμοποιεί το learned heuristic.
    """
    H, W = grid.shape

    dx = gx - x
    dy = gy - y
    abs_dx = abs(dx)
    abs_dy = abs(dy)

    euclid = (dx * dx + dy * dy) ** 0.5
    manhattan = abs_dx + abs_dy
    chebyshev = max(abs_dx, abs_dy)

    # 4-γειτονες
    free_neighbors = 0
    for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
        if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0:
            free_neighbors += 1
    blocked_neighbors = 4 - free_neighbors

    # παράθυρο 3x3
    x_min = max(0, x - 1)
    x_max = min(W, x + 2)
    y_min = max(0, y - 1)
    y_max = min(H, y + 2)
    window = grid[y_min:y_max, x_min:x_max]
    if window.size > 0:
        obstacle_density = float(np.mean(window != 0))
    else:
        obstacle_density = 0.0
    is_near_obstacle = 1.0 if obstacle_density > 0 else 0.0

    norm_x = x / (W - 1)
    norm_y = y / (H - 1)

    return np.array([
        dx,
        dy,
        euclid,
        manhattan,
        chebyshev,
        free_neighbors,
        blocked_neighbors,
        obstacle_density,
        is_near_obstacle,
        norm_x,
        norm_y,
    ], dtype=np.float32)


def build_dataset(num_episodes=200, out_path="planner_dataset_rich.npz"):
    grid = make_grid()
    free_cells = get_free_cells(grid)
    rng = np.random.default_rng(0)

    X_list = []
    y_list = []

    for ep in range(num_episodes):
        s_idx = rng.integers(len(free_cells))
        g_idx = rng.integers(len(free_cells))
        start = free_cells[s_idx]
        goal = free_cells[g_idx]
        if start == goal:
            continue

        path, exp = astar_classic(grid, start, goal)
        if path is None or len(path) == 0:
            continue

        L = len(path)
        for i, (x, y) in enumerate(path):
            remaining = (L - 1) - i  # 0 στο goal
            fx = compute_features(x, y, goal[0], goal[1], grid)
            X_list.append(fx)
            y_list.append(float(remaining))

        print(f"[Episode {ep+1}/{num_episodes}] path len={L}, expansions={exp}, samples={len(X_list)}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    print("Final dataset shape:", X.shape, y.shape)
    np.savez(out_path, X=X, y=y)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build_dataset()
