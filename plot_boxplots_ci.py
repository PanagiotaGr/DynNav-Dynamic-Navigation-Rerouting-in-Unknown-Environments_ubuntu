import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SUMMARY_FILE = "statistical_summary.csv"
OUTPUT_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(SUMMARY_FILE)

# ΜΕΤΡΙΚΕΣ ΠΟΥ ΘΑ ΣΧΕΔΙΑΣΟΥΜΕ
metrics_to_plot = [
    "path_length",
    "time_to_goal",
    "num_expansions",
    "coverage_percent",
    "entropy_reduction"
]

for metric in metrics_to_plot:
    metric_df = df[df["metric"] == metric]

    variants = metric_df["variant"].values
    means = metric_df["mean"].values
    ci = metric_df["ci_95"].values

    x = np.arange(len(variants))

    plt.figure()
    plt.bar(x, means)
    plt.errorbar(x, means, yerr=ci, fmt='o')

    plt.xticks(x, variants, rotation=45)
    plt.ylabel(metric)
    plt.title(f"{metric} (Mean ± 95% CI)")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{metric}_ci.png")
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Saved: {output_path}")

print("✅ All CI boxplots generated.")
