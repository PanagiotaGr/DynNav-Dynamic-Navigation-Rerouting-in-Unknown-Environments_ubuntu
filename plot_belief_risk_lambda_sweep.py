import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "belief_risk_lambda_sweep.csv"

def safe_get(data, candidates):
    for c in candidates:
        if c in data.dtype.names:
            return data[c]
    raise ValueError(f"No matching column found from {candidates}. Available: {data.dtype.names}")

def main():
    data = np.genfromtxt(CSV_PATH, delimiter=",", names=True)

    print("[INFO] Columns found:", data.dtype.names)

    # Τα πραγματικά πεδία του CSV σου:
    # ('lambda', 'found_path', 'total_cost', 'length_cells', 'geometric_length', 'fused_sum', 'fused_mean', 'fused_max')

    lambdas = safe_get(data, ["lambda"])
    # Μπορούμε να δούμε και τα δύο: length σε cells και geometric length
    length_cells = safe_get(data, ["length_cells"])
    geom_length = safe_get(data, ["geometric_length"])
    fused_sum = safe_get(data, ["fused_sum"])
    total_cost = safe_get(data, ["total_cost"])

    # ---- Plot geometric length ----
    plt.figure()
    plt.plot(lambdas, geom_length, marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Geometric path length")
    plt.title("Geometric path length vs risk weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lambda_sweep_geometric_length.png")
    print("[INFO] Saved lambda_sweep_geometric_length.png")

    # ---- Plot length in cells ----
    plt.figure()
    plt.plot(lambdas, length_cells, marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Path length (cells)")
    plt.title("Path length (cells) vs risk weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lambda_sweep_length_cells.png")
    print("[INFO] Saved lambda_sweep_length_cells.png")

    # ---- Plot integrated fused risk ----
    plt.figure()
    plt.plot(lambdas, fused_sum, marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Integrated fused risk")
    plt.title("Integrated fused risk vs risk weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lambda_sweep_fused_risk.png")
    print("[INFO] Saved lambda_sweep_fused_risk.png")

    # ---- Plot total cost ----
    plt.figure()
    plt.plot(lambdas, total_cost, marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Total A* cost")
    plt.title("Total cost vs risk weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lambda_sweep_total_cost.png")
    print("[INFO] Saved lambda_sweep_total_cost.png")

if __name__ == "__main__":
    main()
