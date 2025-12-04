import matplotlib
matplotlib.use("Agg")  # no display, only save

import csv
import matplotlib.pyplot as plt

iters = []
classic_exp = []
learned_exp = []
ratio = []

with open("autotune_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iters.append(int(row["iteration"]))
        classic_exp.append(int(row["classic_expansions"]))
        learned_exp.append(int(row["learned_expansions"]))
        ratio.append(float(row["expansion_ratio"]))

# 1) Expansions plot
plt.figure(figsize=(6, 4))
plt.plot(iters, classic_exp, marker="o", label="Classic A* expansions")
plt.plot(iters, learned_exp, marker="o", label="Learned expansions")
plt.xlabel("Iteration")
plt.ylabel("Expansions")
plt.title("Classic vs Learned A* Expansions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("autotune_expansions.png", dpi=200)

# 2) Expansion ratio
plt.figure(figsize=(6, 4))
plt.plot(iters, ratio, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Expansion Ratio")
plt.title("Learned / Classic Expansion Ratio")
plt.grid(True)
plt.tight_layout()
plt.savefig("autotune_ratio.png", dpi=200)
