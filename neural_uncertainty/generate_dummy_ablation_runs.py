import os
import csv

base_dir = "results/statistical_runs"

variants = {
    "full":        (12.0, 3.0, 520, 93.0, 0.55),
    "no_vo_unc":   (15.0, 4.0, 750, 85.0, 0.31),
    "no_coverage": (14.0, 4.5, 700, 65.0, 0.46),
    "no_priority": (13.0, 3.5, 600, 88.0, 0.48),
    "no_vla":      (13.0, 5.0, 610, 90.0, 0.40),
    "no_online":   (12.5, 4.0, 630, 91.0, 0.44),
}

num_runs = 5

os.makedirs(base_dir, exist_ok=True)

for variant, (pl, ttg, exp, cov, ent) in variants.items():
    var_dir = os.path.join(base_dir, variant)
    os.makedirs(var_dir, exist_ok=True)

    for i in range(num_runs):
        path = os.path.join(var_dir, f"run_{i}.csv")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed",
                "path_length",
                "time_to_goal",
                "num_expansions",
                "num_replans",
                "coverage_percent",
                "entropy_reduction",
                "collision_flag"
            ])
            writer.writerow([
                i,
                pl + 0.1 * i,
                ttg + 0.1 * i,
                exp + 10 * i,
                i,
                cov + 0.1 * i,
                ent,
                0
            ])

print("✅ Dummy ablation runs generated successfully.")
