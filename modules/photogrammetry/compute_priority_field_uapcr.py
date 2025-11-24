#!/usr/bin/env python3
import pandas as pd
import os

# -----------------------------
# Hyperparameters for UAPCR
# -----------------------------
ALPHA = 1.0   # βάρος για μη-κάλυψη (1 - C)
BETA  = 0.5   # βάρος για αβεβαιότητα U
P_MIN = 0.3   # κατώφλι προτεραιότητας για "σημαντικά" κελιά

COVERAGE_CSV = "coverage_grid.csv"
UNCERTAINTY_CSV = "feature_uncertainty_grid.csv"
PRIORITY_CSV = "priority_field.csv"
HIGH_PRIORITY_CSV = "high_priority_cells.csv"


def guess_column(df, keywords, desc):
    """
    Προσπαθεί να βρει στήλη που να περιέχει κάποια από τις λέξεις-κλειδιά.
    """
    cols = list(df.columns)
    for col in cols:
        lower = col.lower()
        if any(k in lower for k in keywords):
            print(f"[INFO] Using '{col}' as {desc} column.")
            return col
    raise ValueError(
        f"Could not find a suitable {desc} column. "
        f"Tried keywords {keywords}. Available columns: {cols}"
    )


def main():
    if not os.path.exists(COVERAGE_CSV):
        raise FileNotFoundError(f"Could not find {COVERAGE_CSV}")
    if not os.path.exists(UNCERTAINTY_CSV):
        raise FileNotFoundError(f"Could not find {UNCERTAINTY_CSV}")

    print(f"Loading coverage grid from {COVERAGE_CSV}...")
    cov_df = pd.read_csv(COVERAGE_CSV)

    print(f"Loading VO uncertainty grid from {UNCERTAINTY_CSV}...")
    unc_df = pd.read_csv(UNCERTAINTY_CSV)

    print(f"[INFO] coverage_grid.csv columns: {list(cov_df.columns)}")
    print(f"[INFO] feature_uncertainty_grid.csv columns: {list(unc_df.columns)}")

    if len(cov_df) != len(unc_df):
        raise ValueError(
            f"Coverage grid has {len(cov_df)} rows but uncertainty grid has {len(unc_df)} rows. "
            "They should match."
        )

    # Δημιουργούμε κοινό index για merge με βάση θέση (όχι cell_x/cell_y).
    cov_df = cov_df.reset_index(drop=True)
    unc_df = unc_df.reset_index(drop=True)
    cov_df["idx"] = cov_df.index
    unc_df["idx"] = unc_df.index

    print("Merging coverage and uncertainty grids on row index...")
    df = cov_df.merge(unc_df, on="idx", suffixes=("_cov", "_unc"))

    # Προσπαθούμε να βρούμε coverage column
    coverage_col = None
    try:
        coverage_col = guess_column(df, ["coverage", "covered"], "coverage")
    except ValueError as e:
        print("[ERROR] Could not auto-detect coverage column.")
        raise e

    # Προσπαθούμε να βρούμε uncertainty column
    uncertainty_col = None
    try:
        uncertainty_col = guess_column(df, ["uncert", "uncertainty", "vo"], "uncertainty")
    except ValueError as e:
        print("[ERROR] Could not auto-detect uncertainty column.")
        raise e

    # Κανονικοποίηση στο [0, 1] για ασφάλεια
    df[coverage_col] = df[coverage_col].clip(0.0, 1.0)
    df[uncertainty_col] = df[uncertainty_col].clip(0.0, 1.0)

    # Υπολογισμός priority field:
    # P(i) = α * (1 - C(i)) + β * U(i)
    print("Computing priority scores...")
    df["priority"] = ALPHA * (1.0 - df[coverage_col]) + BETA * df[uncertainty_col]

    # Αποθήκευση πλήρους priority field
    print(f"Saving full priority field to {PRIORITY_CSV}...")
    df.to_csv(PRIORITY_CSV, index=False)

    # Επιλογή high-priority cells
    high_df = df[df["priority"] >= P_MIN].copy()
    print(f"Selected {len(high_df)} high-priority cells with P >= {P_MIN:.3f}")

    print(f"Saving high-priority cells to {HIGH_PRIORITY_CSV}...")
    high_df.to_csv(HIGH_PRIORITY_CSV, index=False)

    print("=== Priority field computation completed ===")
    print(f"α = {ALPHA}, β = {BETA}, P_min = {P_MIN}")
    print(f"Total cells:       {len(df)}")
    print(f"High-priority:     {len(high_df)}")


if __name__ == "__main__":
    main()
