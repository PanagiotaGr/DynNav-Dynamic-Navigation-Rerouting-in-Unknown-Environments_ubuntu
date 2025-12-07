import os
import glob
import pandas as pd
from scipy.stats import ttest_ind

VARIANT_A = "results/statistical_runs/astar_classic"
VARIANT_B = "results/statistical_runs/astar_learned"

OUTPUT_FILE = "t_test_results.csv"

metrics = [
    "path_length",
    "time_to_goal",
    "num_expansions",
    "entropy_reduction"
]

def load_variant_data(folder):
    all_runs = []
    for csv_file in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(csv_file)
        all_runs.append(df)
    return pd.concat(all_runs, ignore_index=True)

A = load_variant_data(VARIANT_A)
B = load_variant_data(VARIANT_B)

results = []

for metric in metrics:
    a_vals = A[metric].values
    b_vals = B[metric].values

    t_stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)

    results.append({
        "metric": metric,
        "mean_A": a_vals.mean(),
        "mean_B": b_vals.mean(),
        "t_stat": t_stat,
        "p_value": p_val,
        "significant(p<0.05)": p_val < 0.05
    })

df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_FILE, index=False)

print("✅ t-tests completed.")
print(f"✅ Results saved to: {OUTPUT_FILE}")
