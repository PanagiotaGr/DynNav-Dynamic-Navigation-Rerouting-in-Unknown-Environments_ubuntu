import glob
import pandas as pd
import os

SUMMARY_DIR = "results/ids/summary"
OUT_DIR = "results/ids"

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(SUMMARY_DIR, "summary_*.csv")))

if not files:
    raise SystemExit("❌ No summary files found. Run eval_ids_replay.py first.")

dfs = [pd.read_csv(f) for f in files]
df_all = pd.concat(dfs, ignore_index=True)

# Save full table (one row per run)
df_all.to_csv(os.path.join(OUT_DIR, "ids_summary.csv"), index=False)

# Aggregated (paper-ready)
agg = (
    df_all
    .groupby(["mode", "alpha", "N"], dropna=False)
    .agg(
        delay_mean=("delay", "mean"),
        delay_std=("delay", "std"),
        flag_rate_mean=("flag_rate", "mean"),
        trigger_rate_mean=("trigger_rate", "mean"),
        mean_scale_post_mean=("mean_scale_post", "mean"),
        max_scale_post_mean=("max_scale_post", "mean"),
        runs=("seed", "count"),
    )
    .reset_index()
)

agg.to_csv(os.path.join(OUT_DIR, "ids_summary_agg.csv"), index=False)

print("✅ Saved:")
print(" - results/ids/ids_summary.csv        (per-run)")
print(" - results/ids/ids_summary_agg.csv    (aggregated)")
print("\nPreview:")
print(agg.head(10))
