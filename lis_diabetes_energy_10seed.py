"""
Table 3 (Diabetes, Energy Efficiency): 10 independent train/test splits.

Pipeline per split (test set is never used for selecting M):
  split -> StandardScaler on X and on y (train statistics) -> BGD reference
  (eta = 0.1, random normal init, stop when |delta loss| < 1e-9) ->
  M* = capacity with minimum TRAINING loss inside the safe (pre-divergence)
  range of LIS-GD -> number of BGD iterations needed to reach the same
  training loss (iso-loss) -> iteration savings -> test MSE reported once.

Data: Diabetes through scikit-learn (load_diabetes); Energy Efficiency from
ENB2012_data.csv (UCI, columns X1..X8, Y1, Y2; target Y1 = heating load).

Run: python lis_diabetes_energy_10seed.py
"""

import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def LIS(M):
    return [math.log(k, M) - k / M for k in range(2, M + 1)]


def run_bgd(X_train_b, y_train_scaled, seed, eta=0.1, tol=1e-9, max_epoch=100000):
    n = len(X_train_b)
    np.random.seed(seed)
    theta = np.random.randn(X_train_b.shape[1], 1)
    prev_loss = np.mean((y_train_scaled.reshape(-1, 1) - X_train_b @ theta) ** 2)
    loss_history = [prev_loss]
    for epoch in range(1, max_epoch):
        g = 2 / n * X_train_b.T @ (X_train_b @ theta - y_train_scaled.reshape(-1, 1))
        theta = theta - eta * g
        loss = np.mean((y_train_scaled.reshape(-1, 1) - X_train_b @ theta) ** 2)
        loss_history.append(loss)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return theta, loss_history


def find_M_star(X_train_b, y_train_scaled, seed, M_max=400):
    n = len(X_train_b)
    best_M, best_train = None, np.inf
    for test_M in range(10, M_max):
        np.random.seed(seed)
        theta = np.random.randn(X_train_b.shape[1], 1)
        for s in LIS(test_M):
            g = 2 / n * X_train_b.T @ (X_train_b @ theta - y_train_scaled.reshape(-1, 1))
            theta = theta - s * g
        if not np.all(np.isfinite(theta)):
            break
        train_loss = np.mean((X_train_b @ theta - y_train_scaled.reshape(-1, 1)) ** 2)
        if train_loss > 1e6:
            break
        if train_loss < best_train:
            best_M, best_train = test_M, train_loss
        elif train_loss > best_train * 1.5:
            # clearly past the minimum and rising -> stop (Boomerang onset)
            break
    return best_M, best_train


def eval_dataset(loader_fn, seeds, M_max):
    rows = []
    for seed in seeds:
        X_raw, y_raw = loader_fn()
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=seed)
        scX = StandardScaler()
        X_train_scaled = scX.fit_transform(X_train_raw)
        X_test_scaled = scX.transform(X_test_raw)
        scy = StandardScaler()
        y_train_scaled = scy.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_test_scaled = scy.transform(y_test_raw.reshape(-1, 1)).flatten()
        X_train_b = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
        X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]

        theta_bgd, bgd_hist = run_bgd(X_train_b, y_train_scaled, seed)
        bgd_test = np.mean((X_test_b @ theta_bgd - y_test_scaled.reshape(-1, 1)) ** 2)

        M_star, L_star = find_M_star(X_train_b, y_train_scaled, seed, M_max=M_max)

        np.random.seed(seed)
        theta = np.random.randn(X_train_b.shape[1], 1)
        for s in LIS(M_star):
            g = 2 / len(X_train_b) * X_train_b.T @ (X_train_b @ theta - y_train_scaled.reshape(-1, 1))
            theta = theta - s * g
        test_at_star = np.mean((X_test_b @ theta - y_test_scaled.reshape(-1, 1)) ** 2)

        iso_iter = None
        for i, l in enumerate(bgd_hist):
            if l <= L_star:
                iso_iter = i
                break
        lis_iters = M_star - 1
        savings = 100 * (1 - lis_iters / iso_iter) if iso_iter else None

        rows.append(dict(seed=seed, M_star=M_star, train_loss=L_star,
                         test_mse=test_at_star, bgd_test=bgd_test,
                         bgd_iso_iter=iso_iter, lis_iters=lis_iters, savings=savings))
    return pd.DataFrame(rows)


def diabetes_loader():
    d = load_diabetes()
    return d.data, d.target


def energy_loader():
    df = pd.read_csv("ENB2012_data.csv")
    X = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values.astype(float)
    y = df["Y1"].astype(float).values
    return X, y


if __name__ == "__main__":
    import os
    seeds = list(range(10))  # 0..9, ten independent splits
    jobs = [("DIABETES", diabetes_loader, 200)]
    if os.path.exists("ENB2012_data.csv"):
        jobs.append(("ENERGY EFFICIENCY", energy_loader, 300))
    else:
        print("ENB2012_data.csv not found -> skipping Energy Efficiency")

    for name, loader, M_max in jobs:
        print("=" * 60)
        print(f"{name} -- seed robustness (10 splits)")
        print("=" * 60)
        df = eval_dataset(loader, seeds, M_max=M_max)
        print(df.to_string(index=False))
        print("\nSummary (mean +/- std):")
        for col in ["M_star", "train_loss", "test_mse", "bgd_test", "bgd_iso_iter", "savings"]:
            print(f"  {col}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
        print()
