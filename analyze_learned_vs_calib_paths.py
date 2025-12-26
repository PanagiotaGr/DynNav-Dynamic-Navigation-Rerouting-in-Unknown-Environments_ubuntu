import numpy as np
import pandas as pd
import math


def load_1d_grid(csv_path: str, value_col: str):
    """
    Φορτώνει 1D grid από CSV με στήλες: row, col, value_col.
    Υποθέτουμε ότι col=0 και row = 0..H-1.
    Επιστρέφει ένα vector [H] με τις τιμές.
    """
    df = pd.read_csv(csv_path)
    required = ["row", "col", value_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Λείπει η στήλη '{c}' από το {csv_path}.")

    # βρίσκουμε max row
    max_row = int(df["row"].max())
    H = max_row + 1

    vals = np.full(H, np.nan, dtype=float)
    for _, r in df.iterrows():
        i = int(r["row"])
        j = int(r["col"])
        if j != 0:
            # 1D grid, αγνοούμε οτιδήποτε δεν είναι col=0
            continue
        vals[i] = float(r[value_col])

    return vals


def load_path(path_csv: str):
    """
    Φορτώνει path από CSV χωρίς header:
      col0 = row, col1 = col
    """
    arr = np.loadtxt(path_csv, delimiter=",", dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    rows = arr[:, 0]
    cols = arr[:, 1]
    return rows, cols


def path_metrics_1d(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, label: str):
    """
    Υπολογίζει metrics για path σε 1D grid.
      - length (cells)
      - geometric length
      - sum / mean / max values κατά μήκος του path
    """
    assert rows.shape == cols.shape
    n = len(rows)

    length_cells = n

    # geometric length (σε 1D, ουσιαστικά πόσο κινήθηκε πάνω-κάτω)
    geom_len = 0.0
    for k in range(1, n):
        di = rows[k] - rows[k - 1]
        dj = cols[k] - cols[k - 1]
        geom_len += math.sqrt(di * di + dj * dj)

    H = values.shape[0]
    path_vals = []
    for i, j in zip(rows, cols):
        if 0 <= i < H and j == 0:
            path_vals.append(values[i])
        else:
            path_vals.append(np.nan)

    path_vals = np.array(path_vals, dtype=float)
    valid = path_vals[~np.isnan(path_vals)]
    if valid.size == 0:
        v_sum = np.nan
        v_mean = np.nan
        v_max = np.nan
    else:
        v_sum = float(valid.sum())
        v_mean = float(valid.mean())
        v_max = float(valid.max())

    metrics = {
        "label": label,
        "length_cells": length_cells,
        "geometric_length": geom_len,
        "value_sum": v_sum,
        "value_mean": v_mean,
        "value_max": v_max,
    }
    return metrics


def main():
    # 1D grids
    learned_grid_csv = "learned_uncertainty_grid.csv"
    calib_grid_csv = "calib_learned_uncertainty_grid.csv"

    print("[INFO] Loading 1D grids...")
    learned_vals = load_1d_grid(learned_grid_csv, "learned_sigma")
    calib_vals = load_1d_grid(calib_grid_csv, "calib_sigma")

    print("[INFO] learned_vals shape:", learned_vals.shape)
    print("[INFO] calib_vals shape:", calib_vals.shape)

    # Paths
    path_plain_csv = "belief_risk_path_case_learned_plain.csv"
    path_calib_csv = "belief_risk_path_case_learned_calib.csv"

    print("[INFO] Loading paths...")
    rows_plain, cols_plain = load_path(path_plain_csv)
    rows_calib, cols_calib = load_path(path_calib_csv)
    print("[INFO] Plain path length:", len(rows_plain))
    print("[INFO] Calib path length:", len(rows_calib))

    # Metrics
    m_plain_on_plain = path_metrics_1d(
        rows_plain, cols_plain, learned_vals, label="plain_path_on_plain_sigma"
    )
    m_plain_on_calib = path_metrics_1d(
        rows_plain, cols_plain, calib_vals, label="plain_path_on_calib_sigma"
    )

    m_calib_on_plain = path_metrics_1d(
        rows_calib, cols_calib, learned_vals, label="calib_path_on_plain_sigma"
    )
    m_calib_on_calib = path_metrics_1d(
        rows_calib, cols_calib, calib_vals, label="calib_path_on_calib_sigma"
    )

    # Εκτύπωση
    def pretty_print(m):
        print(f"\n=== {m['label']} ===")
        print(f"Length (cells): {m['length_cells']:.0f}")
        print(f"Geometric length: {m['geometric_length']:.3f}")
        print(
            f"Value: sum={m['value_sum']:.3f}, "
            f"mean={m['value_mean']:.3f}, "
            f"max={m['value_max']:.3f}"
        )

    pretty_print(m_plain_on_plain)
    pretty_print(m_plain_on_calib)
    pretty_print(m_calib_on_plain)
    pretty_print(m_calib_on_calib)

    # CSV output
    rows_out = [
        m_plain_on_plain,
        m_plain_on_calib,
        m_calib_on_plain,
        m_calib_on_calib,
    ]
    df_out = pd.DataFrame(rows_out)
    out_csv = "learned_vs_calib_path_metrics.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved metrics to {out_csv}")


if __name__ == "__main__":
    main()
