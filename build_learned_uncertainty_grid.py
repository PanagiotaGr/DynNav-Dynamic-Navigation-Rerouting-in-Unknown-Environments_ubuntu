import numpy as np
import pandas as pd
import torch

from train_drift_uncertainty_net import (
    CSV_PATH,
    MODEL_PATH,
    NORM_STATS_PATH,
    UncertaintyMLP,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_stats():
    """
    Φορτώνει το εκπαιδευμένο μοντέλο drift-uncertainty
    και τα normalization statistics (x_mean, x_std, y_mean, y_std).
    """
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    input_dim = ckpt["input_dim"]
    feature_cols = ckpt["feature_cols"]
    target_col = ckpt["target_col"]

    model = UncertaintyMLP(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    stats = np.load(NORM_STATS_PATH)
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    return model, feature_cols, target_col, x_mean, x_std, y_mean, y_std


def main():
    print("[INFO] Loading model and stats...")
    model, feature_cols, target_col, x_mean, x_std, y_mean, y_std = load_model_and_stats()

    print("[INFO] Loading drift dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    num_df = df.select_dtypes(include=[np.number])

    # Βεβαιωνόμαστε ότι όλες οι feature_cols υπάρχουν
    for c in feature_cols:
        if c not in num_df.columns:
            raise ValueError(f"Feature column '{c}' λείπει από το drift_dataset.csv.")

    X = num_df[feature_cols].to_numpy(dtype=np.float32)

    # Δημιουργούμε synthetic indices:
    # κάθε δείγμα drift γίνεται ένα κελί σε 1D "γραμμή"
    # row = index δείγματος, col = 0
    n = len(num_df)
    rows = np.arange(n, dtype=int)
    cols = np.zeros(n, dtype=int)

    # Normalize inputs με τα ίδια stats όπως στο training
    Xn = (X - x_mean) / x_std

    print("[INFO] Running model to predict sigma (uncertainty)...")
    with torch.no_grad():
        xb = torch.from_numpy(Xn).to(DEVICE)
        mean_n, log_std = model(xb)
        log_std = log_std.cpu().numpy()

    # Unnormalize std στο original scale του target
    std_norm = np.exp(log_std)          # std στο normalized space
    std_orig = std_norm * y_std         # scale με y_std για original space
    learned_sigma = std_orig.reshape(-1)

    # Φτιάχνουμε 1D grid [H, W] με H = n, W = 1
    H = n
    W = 1
    grid = np.full((H, W), np.nan, dtype=float)

    for r, c, s in zip(rows, cols, learned_sigma):
        if 0 <= r < H and 0 <= c < W:
            grid[r, c] = float(s)

    # Flatten σε CSV με row, col, learned_sigma
    out_rows = []
    for r in range(H):
        for c in range(W):
            s = grid[r, c]
            # Αν για κάποιο λόγο είναι NaN, το αφήνουμε έτσι·
            # ο planner μπορεί να το χειριστεί (π.χ. unc=0 αν NaN).
            out_rows.append(
                {
                    "row": r,
                    "col": c,
                    "learned_sigma": s,
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_csv = "learned_uncertainty_grid.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved learned uncertainty grid to {out_csv}")
    print("[INFO] Grid shape:", H, "x", W)


if __name__ == "__main__":
    main()
