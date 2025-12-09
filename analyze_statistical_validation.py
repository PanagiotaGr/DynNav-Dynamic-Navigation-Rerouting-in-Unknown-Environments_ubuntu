import os
import glob
import pandas as pd
import numpy as np

ROOT_DIR = "results/statistical_runs"
OUTPUT_FILE = "statistical_summary.csv"

records = []

for variant in sorted(os.listdir(ROOT_DIR)):
    variant_path = os.path.join(ROOT_DIR, variant)

    if not os.path.isdir(variant_path):
        continue

    all_runs = []

    for csv_file in glob.glob(os.path.join(variant_path, "*.csv")):
        df = pd.read_csv(csv_file)
        all_runs.append(df)

    if len(all_runs) < 5:
        print(f"⚠️  WARNING: {variant} has less than 5 runs!")
        continue

    data = pd.concat(all_runs, ignore_index=True)

    metrics = [
        "path_length",
        "time_to_goal",
        "num_expansions",
        "num_replans",
        "coverage_percent",
        "entropy_reduction",
        "collision_flag"
    ]

    N = len(data)

    for metric in metrics:
        values = data[metric].values

        mean = np.mean(values)
        std = np.std(values, ddof=1)  # sample std
        ci_95 = 1.96 * std / np.sqrt(N)

        records.append({
            "variant": variant,
            "metric": metric,
            "N": N,
            "mean": mean,
            "std": std,
            "ci_95": ci_95,
            "ci_low": mean - ci_95,
            "ci_high": mean + ci_95
        })

summary_df = pd.DataFrame(records)
summary_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Statistical validation completed.")
print(f"✅ Results saved to: {OUTPUT_FILE}")
