import csv

from astar_learned_heuristic import (
    make_grid,
    astar_classic,
    astar_learned,
    LearnedHeuristicWrapper,
    compute_true_remaining_cost,
)
from heuristic_logger import HeuristicLogger
from online_update_heuristic import online_update


def run_one_iteration(iter_idx: int,
                      model_path: str = "heuristic_net_rich.pt",
                      log_path: str = "heuristic_logs.npz"):
    grid = make_grid()
    start = (1, 1)
    goal = (35, 35)

    print(f"[Iter {iter_idx}] Running classic A*...")
    path_c, exp_c = astar_classic(grid, start, goal)
    print(f"[Iter {iter_idx}] classic: path length={len(path_c)}, expansions={exp_c}")

    true_cost_map = compute_true_remaining_cost(path_c)

    print(f"[Iter {iter_idx}] Running learned A* with logging...")
    logger = HeuristicLogger(log_path, true_cost_map=true_cost_map)
    lh = LearnedHeuristicWrapper(model_path=model_path, logger=logger)
    path_l, exp_l = astar_learned(grid, start, goal, lh)
    print(f"[Iter {iter_idx}] learned: path length={len(path_l)}, expansions={exp_l}")

    logger.save()

    res = {
        "iteration": iter_idx,
        "classic_path_len": len(path_c),
        "classic_expansions": exp_c,
        "learned_path_len": len(path_l),
        "learned_expansions": exp_l,
    }
    res["expansion_ratio"] = res["learned_expansions"] / res["classic_expansions"]
    res["delta_path"] = res["learned_path_len"] - res["classic_path_len"]
    return res


def main():
    model_path = "heuristic_net_rich.pt"
    log_path = "heuristic_logs.npz"
    out_csv = "autotune_results.csv"
    num_iters = 10

    rows = []

    for it in range(num_iters):
        print("\n" + "=" * 50)
        print(f"[AUTO] Iteration {it}")
        print("=" * 50)

        res = run_one_iteration(it, model_path=model_path, log_path=log_path)
        rows.append(res)

        print(f"[AUTO] Online update after iteration {it}...")
        online_update(model_path=model_path, log_path=log_path,
                      epochs=3, lr=1e-4, batch_size=128)

    fieldnames = [
        "iteration",
        "classic_path_len",
        "classic_expansions",
        "learned_path_len",
        "learned_expansions",
        "expansion_ratio",
        "delta_path",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[AUTO] Saved autotune results to {out_csv}")


if __name__ == "__main__":
    main()
