import numpy as np
import pandas as pd
from typing import Tuple


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardization: (x - mean) / std
    Επιστρέφει X_norm, mean, std
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid div by zero
    X_norm = (X - mean) / std
    return X_norm, mean, std


def train_ridge_regression(X: np.ndarray, y: np.ndarray, lam: float = 1e-3):
    """
    Ridge Regression:
    w = (X^T X + lam I)^(-1) X^T y
    bias = mean of y - mean(X)·w
    Επιστρέφει w, b
    """
    n_samples, n_features = X.shape
    XTX = X.T @ X
    I = np.eye(n_features)
    w = np.linalg.inv(XTX + lam * I) @ (X.T @ y)

    # bias από το original y και X (non-normalized)
    # αλλά εδώ έχουμε ήδη standardized X, οπότε κρατάμε bias=0
    b = 0.0
    return w, b


def evaluate_model(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    y_pred = X @ w + b
    mse = np.mean((y_pred - y) ** 2)
    # Pearson correlation
    y_mean = y.mean()
    y_std = y.std()
    yp_mean = y_pred.mean()
    yp_std = y_pred.std()
    if y_std < 1e-8 or yp_std < 1e-8:
        corr = 0.0
    else:
        corr = np.mean((y - y_mean) * (y_pred - yp_mean)) / (y_std * yp_std)
    return mse, corr


if __name__ == "__main__":
    dataset_csv = "drift_dataset.csv"
    model_path = "drift_model.npz"

    print(f"[DRIFT-TRAIN] Loading dataset: {dataset_csv}")
    df = pd.read_csv(dataset_csv)

    # Features & target
    feature_cols = ["entropy", "local_uncertainty", "speed"]
    target_col = "drift"

    X = df[feature_cols].to_numpy().astype(float)
    y = df[target_col].to_numpy().astype(float)

    # Split σε train / test (80 / 20)
    n = len(X)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    split = int(0.8 * n)
    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"[DRIFT-TRAIN] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Standardize features
    X_train_norm, mean, std = standardize(X_train)
    X_test_norm = (X_test - mean) / std

    # Train ridge regression
    print("[DRIFT-TRAIN] Training ridge regression model...")
    w, b = train_ridge_regression(X_train_norm, y_train, lam=1e-3)

    # Evaluate
    train_mse, train_corr = evaluate_model(X_train_norm, y_train, w, b)
    test_mse, test_corr = evaluate_model(X_test_norm, y_test, w, b)

    print("\n[DRIFT-TRAIN] === Results ===")
    print(f"Train MSE: {train_mse:.6f}, Train corr: {train_corr:.4f}")
    print(f"Test  MSE: {test_mse:.6f}, Test  corr: {test_corr:.4f}")

    # Save model
    np.savez(
        model_path,
        w=w,
        b=b,
        mean=mean,
        std=std,
        feature_cols=np.array(feature_cols),
    )
    print(f"[DRIFT-TRAIN] Saved model to: {model_path}")
