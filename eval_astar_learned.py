import csv
from astar_learned_heuristic import (
    make_grid,
    astar_classic,
    astar_learned,
    LearnedHeuristicWrapper,
)


def generate_demo_world():
    grid = make_grid()
    start = (1, 1)
    goal = (35, 35)
    return grid, start, goal


def run_single(seed: int):
    grid, start, goal = generate_demo_world()

    path_c, exp_c = astar_classic(grid, start, goal)

    lh = LearnedHeuristicWrapper(model_path="heuristic_net_rich.pt")
    path_l, exp_l = astar_learned(grid, start, goal, lh)

    return {
        "seed": seed,
        "classic_path_len": len(path_c),
        "classic_expansions": exp_c,
        "learned_path_len": len(path_l),
        "learned_expansions": exp_l,
    }


def main():
    out_csv = "astar_eval_results.csv"
    num_trials = 10

    rows = []
    for seed in range(num_trials):
        print(f"Running seed {seed}...")
        res = run_single(seed)
        res["expansion_ratio"] = res["learned_expansions"] / res["classic_expansions"]
        res["delta_path"] = res["learned_path_len"] - res["classic_path_len"]
        rows.append(res)

    fieldnames = [
        "seed",
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

    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
