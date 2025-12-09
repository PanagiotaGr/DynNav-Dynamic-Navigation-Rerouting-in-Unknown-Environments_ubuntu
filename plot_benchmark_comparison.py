import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results.csv")

metrics = [
    "Path_length",
    "Time_to_goal",
    "Number_of_expansions",
    "Replans",
    "Coverage_percent",
    "Collision_rate",
    "Entropy_reduction"
]

for metric in metrics:
    plt.figure()
    plt.bar(df["Algorithm"], df[metric])
    plt.title(metric.replace("_", " "))
    plt.xlabel("Algorithm")
    plt.ylabel(metric.replace("_", " "))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"plot_{metric.lower()}.png")
    plt.close()
