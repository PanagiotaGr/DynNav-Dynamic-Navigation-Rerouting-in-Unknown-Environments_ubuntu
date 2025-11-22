import csv
import matplotlib.pyplot as plt
from collections import defaultdict


def load_sweep_results(csv_path="planner_sweep_results.csv"):
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                {
                    "frontlap": float(row["frontlap"]),
                    "sidelap": float(row["sidelap"]),
                    "step_x_m": float(row["step_x_m"]),
                    "step_y_m": float(row["step_y_m"]),
                    "num_waypoints": int(row["num_waypoints"]),
                    "flight_time_s": float(row["flight_time_s"]),
                    "flight_time_min": float(row["flight_time_min"]),
                }
            )
    return results


def plot_num_waypoints_vs_frontlap(results):
    """
    Plot: frontlap vs αριθμός waypoints
    για διαφορετικά sidelap (διαφορετική καμπύλη κάθε φορά).
    """
    # group by sidelap
    by_sidelap = defaultdict(list)
    for r in results:
        by_sidelap[r["sidelap"]].append(r)

    plt.figure()
    for sidelap, rows in sorted(by_sidelap.items()):
        rows_sorted = sorted(rows, key=lambda r: r["frontlap"])
        frontlaps = [r["frontlap"] for r in rows_sorted]
        num_wps = [r["num_waypoints"] for r in rows_sorted]
        plt.plot(frontlaps, num_wps, marker="o", label=f"sidelap={sidelap:.1f}")

    plt.xlabel("Frontlap")
    plt.ylabel("Number of waypoints")
    plt.title("Number of waypoints vs frontlap (for different sidelap values)")
    plt.grid(True)
    plt.legend()


def plot_flight_time_vs_frontlap(results):
    """
    Plot: frontlap vs χρόνος πτήσης (min)
    για διαφορετικά sidelap.
    """
    by_sidelap = defaultdict(list)
    for r in results:
        by_sidelap[r["sidelap"]].append(r)

    plt.figure()
    for sidelap, rows in sorted(by_sidelap.items()):
        rows_sorted = sorted(rows, key=lambda r: r["frontlap"])
        frontlaps = [r["frontlap"] for r in rows_sorted]
        flight_time_min = [r["flight_time_min"] for r in rows_sorted]
        plt.plot(frontlaps, flight_time_min, marker="o", label=f"sidelap={sidelap:.1f}")

    plt.xlabel("Frontlap")
    plt.ylabel("Flight time (min)")
    plt.title("Estimated flight time vs frontlap (for different sidelap values)")
    plt.grid(True)
    plt.legend()


def main():
    results = load_sweep_results("planner_sweep_results.csv")
    print(f"Loaded {len(results)} configurations from planner_sweep_results.csv")

    plot_num_waypoints_vs_frontlap(results)
    plot_flight_time_vs_frontlap(results)

    plt.show()


if __name__ == "__main__":
    main()
