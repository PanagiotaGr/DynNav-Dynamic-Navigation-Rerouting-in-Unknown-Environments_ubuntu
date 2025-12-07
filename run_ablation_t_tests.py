import os
import glob
import pandas as pd
from scipy.stats import ttest_ind

BASE_DIR = "results/statistical_runs"
FULL_DIR = os.path.join(BASE_DIR, "full")

ABLATION_DIRS = {
    "no_vo_unc": "Full vs No-VO Uncertainty",
    "no_coverage": "Full vs No-Coverage",
    "no_priority": "Full vs No-Priority",
    "no_vla": "Full vs No-VLA",
    "no_online": "Full vs No-Online"
}

OUTPUT_FILE = "ablation_t_test_results.csv"

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

FULL = load_variant_data(FULL_DIR)

results = []

for ablation_folder, label in ABLATION_DIRS.items():
    ablation_path = os.path.join(BASE_DIR, ablation_folder)
    ABL = load_variant_data(ablation_path)

    for metric in metrics:
        full_vals = FULL[metric].values
        abl_vals = ABL[metric].values

        t_stat, p_val = ttest_ind(full_vals, abl_vals, equal_var=False)

        results.append({
            "comparison": label,
            "metric": metric,
            "mean_full": full_vals.mean(),
            "mean_ablation": abl_vals.mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "significant(p<0.05)": p_val < 0.05
        })

df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_FILE, index=False)

print("✅ Ablation t-tests completed.")
print(f"✅ Results saved to: {OUTPUT_FILE}")
