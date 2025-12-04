import argparse
import heapq

import numpy as np
import matplotlib.pyplot as plt
import torch

from learned_heuristic import HeuristicNet
from heuristic_logger import HeuristicLogger


# ---------------- Grid world ----------------

def make_grid():
    """
    Απλό 40x40 grid με κάποια εμπόδια.
    0 = free, 1 = obstacle.
    """
    H, W = 40, 40
    grid = np.zeros((H, W), dtype=np.int32)

    grid[5, 2:35] = 1
    grid[15, 5:30] = 1
    grid[10:30, 20] = 1
    grid[25:30, 8:15] = 1

    return grid


# ---------------- Classic A* ----------------

def astar_classic(grid, start, goal):
    H, W = grid.shape

    def in_bounds(x, y):
        return 0 <= x < W and 0 <= y < H

    def is_free(x, y):
        return grid[y, x] == 0

    def heuristic(p, q):
        (x1, y1), (x2, y2) = p, q
        dx = x2 - x1
        dy = y2 - y1
        return (dx * dx + dy * dy) ** 0.5

    neighbors_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    g = {start: 0.0}
    parent = {start: None}

    f_start = heuristic(start, goal)
    open_pq = [(f_start, 0.0, start)]
    closed = set()
    expansions = 0

    while open_pq:
        f_curr, g_curr, (x, y) = heapq.heappop(open_pq)
        curr = (x, y)
        if curr in closed:
            continue
        closed.add(curr)
        expansions += 1

        if curr == goal:
            path = []
            node = curr
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, expansions

        for dx, dy in neighbors_4:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            if not is_free(nx, ny):
                continue

            neigh = (nx, ny)
            tentative_g = g_curr + 1.0
            if neigh not in g or tentative_g < g[neigh]:
                g[neigh] = tentative_g
                parent[neigh] = curr
                h_val = heuristic(neigh, goal)
                f_val = tentative_g + h_val
                heapq.heappush(open_pq, (f_val, tentative_g, neigh))

    return None, expansions


# ---------------- True remaining cost map ----------------

def compute_true_remaining_cost(path):
    """
    Από το πλήρες path (λίστα (x,y)), φτιάχνει dict:
      node -> remaining steps μέχρι goal.
    """
    rem = {}
    L = len(path)
    for i, node in enumerate(path):
        rem[node] = L - i - 1  # 0 στο goal
    return rem


# ---------------- Learned heuristic wrapper ----------------

class LearnedHeuristicWrapper:
    def __init__(self, model_path="heuristic_net_rich.pt", logger=None):
        self.device = torch.device("cpu")
        self.model = HeuristicNet(input_dim=11)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.logger = logger  # μπορεί να είναι None

    def _compute_features(self, node, goal, grid):
        import numpy as np
        x, y = node
        gx, gy = goal
        H, W = grid.shape

        dx = gx - x
        dy = gy - y
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        euclid = (dx * dx + dy * dy) ** 0.5
        manhattan = abs_dx + abs_dy
        chebyshev = max(abs_dx, abs_dy)

        free_neighbors = 0
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0:
                free_neighbors += 1
        blocked_neighbors = 4 - free_neighbors

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

        features = np.array([
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

        return features

    def h(self, node, goal, grid):
        """
        Επιστρέφει την εκτίμηση heuristic για το node,
        και αν υπάρχει logger, κάνει record(features, node) με ground truth.
        """
        features = self._compute_features(node, goal, grid)

        with torch.no_grad():
            x_t = torch.from_numpy(features).to(self.device)
            out = self.model(x_t)
        h_val = float(out.item())

        if self.logger is not None:
            self.logger.record(features, node)

        return h_val


# ---------------- Learned A* ----------------

def astar_learned(grid, start, goal, learned_h):
    H, W = grid.shape

    def in_bounds(x, y):
        return 0 <= x < W and 0 <= y < H

    def is_free(x, y):
        return grid[y, x] == 0

    neighbors_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    g = {start: 0.0}
    parent = {start: None}

    f_start = learned_h.h(start, goal, grid)
    open_pq = [(f_start, 0.0, start)]
    closed = set()
    expansions = 0

    while open_pq:
        f_curr, g_curr, (x, y) = heapq.heappop(open_pq)
        curr = (x, y)
        if curr in closed:
            continue
        closed.add(curr)
        expansions += 1

        if curr == goal:
            path = []
            node = curr
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, expansions

        for dx, dy in neighbors_4:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            if not is_free(nx, ny):
                continue

            neigh = (nx, ny)
            tentative_g = g_curr + 1.0
            if neigh not in g or tentative_g < g[neigh]:
                g[neigh] = tentative_g
                parent[neigh] = curr
                h_val = learned_h.h(neigh, goal, grid)
                f_val = tentative_g + h_val
            # push
                heapq.heappush(open_pq, (f_val, tentative_g, neigh))

    return None, expansions


# ---------------- Visualization ----------------

def plot_paths(grid, path_c, path_l, start, goal):
    H, W = grid.shape
    img = np.ones((H, W, 3), dtype=np.float32)
    img[grid == 1] = [0.0, 0.0, 0.0]

    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="lower")

    xs_c = [p[0] for p in path_c]
    ys_c = [p[1] for p in path_c]
    xs_l = [p[0] for p in path_l]
    ys_l = [p[1] for p in path_l]

    plt.plot(xs_c, ys_c, 'g-', label="A* classic")
    plt.plot(xs_l, ys_l, 'r--', label="A* learned")

    plt.scatter([start[0]], [start[1]], c='blue', marker='o', label="start")
    plt.scatter([goal[0]], [goal[1]], c='yellow', marker='*', label="goal")

    plt.legend()
    plt.title("Classic vs Learned A*")
    plt.tight_layout()
    plt.show()


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    grid = make_grid()
    start = (1, 1)
    goal = (35, 35)

    print("Running A* classic...")
    path_c, exp_c = astar_classic(grid, start, goal)
    print(f"classic: path length={len(path_c)}, expansions={exp_c}")

    # ground-truth remaining cost από classic path
    true_cost_map = compute_true_remaining_cost(path_c)

    print("Loading learned heuristic with logger...")
    logger = HeuristicLogger("heuristic_logs.npz", true_cost_map=true_cost_map)
    lh = LearnedHeuristicWrapper(model_path="heuristic_net_rich.pt", logger=logger)

    print("Running A* with learned heuristic...")
    path_l, exp_l = astar_learned(grid, start, goal, lh)
    print(f"learned: path length={len(path_l)}, expansions={exp_l}")

    logger.save()

    if args.demo:
        plot_paths(grid, path_c, path_l, start, goal)


if __name__ == "__main__":
    main()
