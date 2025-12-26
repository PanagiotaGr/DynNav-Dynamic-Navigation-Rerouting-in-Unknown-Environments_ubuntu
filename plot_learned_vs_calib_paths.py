import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "learned_vs_calib_path_metrics.csv"

def main():
    # Διαβάζουμε το CSV με ονόματα στηλών
    data = np.genfromtxt(
        CSV_PATH,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    print("[INFO] Columns:", data.dtype.names)

    names = data.dtype.names

    # Προσπαθούμε να εντοπίσουμε τη στήλη που περιγράφει το σενάριο (case / name / scenario)
    scenario_col = None
    for cand in ["scenario", "case", "name", "label"]:
        if cand in names:
            scenario_col = cand
            break

    if scenario_col is None:
        raise ValueError(
            f"No scenario/name column found. Available: {names}"
        )

    scenarios = data[scenario_col]

    # Βρίσκουμε στήλες για sum/mean/max risk
    def safe_get(candidates):
        for c in candidates:
            if c in names:
                return data[c], c
        raise ValueError(
            f"No matching column found from {candidates}. Available: {names}"
        )

    risk_sum, sum_name = safe_get(["value_sum", "sum_value", "sum", "risk_sum"])
    risk_mean, mean_name = safe_get(["value_mean", "mean_value", "mean", "risk_mean"])
    risk_max, max_name = safe_get(["value_max", "max_value", "max", "risk_max"])

    print(f"[INFO] Using columns: sum={sum_name}, mean={mean_name}, max={max_name}")

    # Φτιάχνουμε x-άξονα με index 0..N-1
    x = np.arange(len(scenarios))

    # ---- Plot sum ----
    plt.figure()
    plt.bar(x, risk_sum)
    plt.xticks(x, scenarios, rotation=20, ha="right")
    plt.ylabel("Integrated risk (sum)")
    plt.title("Learned vs calibrated uncertainty: integrated risk")
    plt.tight_layout()
    plt.savefig("learned_vs_calib_sum.png")
    print("[INFO] Saved learned_vs_calib_sum.png")

    # ---- Plot mean ----
    plt.figure()
    plt.bar(x, risk_mean)
    plt.xticks(x, scenarios, rotation=20, ha="right")
    plt.ylabel("Mean risk")
    plt.title("Learned vs calibrated uncertainty: mean risk")
    plt.tight_layout()
    plt.savefig("learned_vs_calib_mean.png")
    print("[INFO] Saved learned_vs_calib_mean.png")

    # ---- Plot max ----
    plt.figure()
    plt.bar(x, risk_max)
    plt.xticks(x, scenarios, rotation=20, ha="right")
    plt.ylabel("Max risk")
    plt.title("Learned vs calibrated uncertainty: max risk")
    plt.tight_layout()
    plt.savefig("learned_vs_calib_max.png")
    print("[INFO] Saved learned_vs_calib_max.png")


if __name__ == "__main__":
    main()
