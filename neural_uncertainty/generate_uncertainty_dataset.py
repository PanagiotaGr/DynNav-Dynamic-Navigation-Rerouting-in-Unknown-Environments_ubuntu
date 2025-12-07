import csv
import random
import numpy as np

random.seed(0)
np.random.seed(0)

N = 2000

with open("uncertainty_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "inliers", "drift", "lin_vel",
        "ang_vel", "entropy", "target_uncertainty"
    ])

    for _ in range(N):
        inliers = random.randint(50, 150)
        drift = random.uniform(0.01, 0.25)
        lin_vel = random.uniform(0.1, 0.5)
        ang_vel = random.uniform(0.01, 0.2)
        entropy = random.uniform(0.1, 0.9)

        # Synthetic ground-truth uncertainty model
        target_unc = (
            0.003 * (150 - inliers)
            + 0.9 * drift
            + 0.4 * ang_vel
            + 0.2 * entropy
        )

        writer.writerow([
            inliers, drift, lin_vel, ang_vel, entropy, target_unc
        ])

print("✅ uncertainty_dataset.csv created.")
