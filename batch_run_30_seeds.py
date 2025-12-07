import os
import subprocess
import time
import csv
import random

# ==============================
# CONFIG
# ==============================

PLANNER_SCRIPT = "eval_astar_learned.py"  
OUTPUT_DIR = "results/statistical_runs/astar_learned"
NUM_RUNS = 30
TIMEOUT = 120    # seconds per run (ασφάλεια)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# MAIN LOOP
# ==============================

for seed in range(NUM_RUNS):
    print(f"\n🚀 Running seed {seed}/{NUM_RUNS - 1}")

    output_csv = os.path.join(OUTPUT_DIR, f"run_{seed}.csv")

    # Κλήση του planner με seed
    cmd = [
        "python",
        PLANNER_SCRIPT,
        "--seed", str(seed),
        "--output", output_csv
    ]

    try:
        start_time = time.time()

        subprocess.run(cmd, timeout=TIMEOUT, check=True)

        elapsed = time.time() - start_time
        print(f"✅ Completed seed {seed} in {elapsed:.2f}s")

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT at seed {seed}")
    except subprocess.CalledProcessError:
        print(f"❌ ERROR at seed {seed}")

print("\n✅ Batch execution finished.")
