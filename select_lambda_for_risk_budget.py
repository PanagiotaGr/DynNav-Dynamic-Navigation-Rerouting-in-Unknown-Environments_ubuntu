import numpy as np

CSV_PATH = "belief_risk_lambda_sweep.csv"

def select_lambda_for_budget(risk_budget: float):
    data = np.genfromtxt(CSV_PATH, delimiter=",", names=True)

    lambdas = data["lambda"]
    fused_sum = data["fused_sum"]
    total_cost = data["total_cost"]

    # βρίσκουμε όλα τα λ που σέβονται το risk budget
    valid_idx = np.where(fused_sum <= risk_budget)[0]

    if len(valid_idx) == 0:
        print(f"[WARN] No λ satisfies fused_sum <= {risk_budget:.3f}")
        # παίρνουμε αυτό με το μικρότερο fused_sum για info
        best_idx = np.argmin(fused_sum)
        print(
            f"[INFO] Closest: λ={lambdas[best_idx]:.3f}, "
            f"fused_sum={fused_sum[best_idx]:.3f}, total_cost={total_cost[best_idx]:.3f}"
        )
    else:
        # από αυτούς που σέβονται το budget, πάρε τον μικρότερο λ (πιο "κλασικό" A*)
        best_idx = valid_idx[0]
        print(
            f"[INFO] Selected λ={lambdas[best_idx]:.3f} "
            f"for risk_budget={risk_budget:.3f} "
            f"(fused_sum={fused_sum[best_idx]:.3f}, total_cost={total_cost[best_idx]:.3f})"
        )

def main():
    budgets = [30.0, 25.0, 20.0, 18.0, 15.0]  # βάλε όποια thresholds σε ενδιαφέρουν
    for R in budgets:
        select_lambda_for_budget(R)

if __name__ == "__main__":
    main()
