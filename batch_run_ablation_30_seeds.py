import os
import subprocess
import time

# ==============================
# CONFIG
# ==============================

PLANNER_SCRIPT = "eval_astar_learned.py"
BASE_OUTPUT_DIR = "results/statistical_runs"
NUM_RUNS = 30
TIMEOUT = 120  # seconds per run

ABLATION_VARIANTS = {
    "full": [],
    "no_vo_unc": ["--disable_vo_uncertainty"],
    "no_coverage": ["--disable_coverage"],
    "no_priority": ["--disable_priority"],
    "no_vla": ["--disable_vla"],
    "no_online": ["--disable_online"]
}

# ==============================
# MAIN LOOP
# ==============================

for variant_name, variant_flags in ABLATION_VARIANTS.items():

    variant_dir = os.path.join(BASE_OUTPUT_DIR, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    print(f"\n🔥 Running ablation variant: {variant_name}\n")

    for seed in range(NUM_RUNS):
        print(f"🚀 [{variant_name}] Seed {seed}/{NUM_RUNS - 1}")

        output_csv = os.path.join(variant_dir, f"run_{seed}.csv")

        cmd = [
            "python",
            PLANNER_SCRIPT,
            "--seed", str(seed),
            "--output", output_csv
        ] + variant_flags

        try:
            start_time = time.time()
            subprocess.run(cmd, timeout=TIMEOUT, check=True)
            elapsed = time.time() - start_time
            print(f"✅ Completed in {elapsed:.2f}s")

        except subprocess.TimeoutExpired:
            print(f"❌ TIMEOUT at seed {seed} ({variant_name})")
        except subprocess.CalledProcessError:
            print(f"❌ ERROR at seed {seed} ({variant_name})")

print("\n✅ Ablation 30-seeds batch completed.")
