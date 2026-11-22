"""
LIS-GD -- Section 4.1, Advertising dataset (Table 8).

Data: classic ISLR "Advertising.csv" (TV, Radio, Newspaper -> Sales),
n = 200, d = 3 features + intercept.

Same protocol as the synthetic experiment (Ek A.2): plain linear
regression, full-batch gradient descent, canonical (single-profile)
LIS-GD, target d_beta <= 1e-8 tolerance to the OLS solution.

Paper's reported results (Table 8 / Section 4.1):
    Candidate M (tolerance-reaching)      = 43
    Single-run M used for the headline
    comparison against BGD                = 49
    BGD reaches test MSE = 2.908 in 71 iterations
    LIS-GD (M=49) reaches test MSE = 2.908 (identical)
    No divergence observed for any M <= 2000 tested

NOTE 1: the manuscript does not state the exact train/test split it
used for the test-MSE figures. This script uses an 80/20 split with
random_state=42 (the project's standard seed) as the most natural
default -- if your reproduction of Table 8 used a different split,
change `test_size`/`random_state` below to match your own protocol.

NOTE 2: features are standardized (z-score, fit on train only) before
fitting. This is required for LIS-GD/BGD to be numerically stable at
all: on raw TV/Radio/Newspaper scale, lambda_max of H=(2/m)X^T X is
~6.0e4 (verified numerically), which diverges under any LIS-GD or BGD
schedule with I_extra/eta of order 0.1-0.5. Standardizing brings
lambda_max down to ~2.7 (same order as the Section 3.2 synthetic
experiment's 4.1301), consistent with the paper's claim of "no
divergence up to M=2000" for this dataset. Sales (target) is left in
its original units so the reported MSE is directly comparable.

Requires: Advertising.csv in the same directory (TV,Radio,Newspaper,Sales
columns; an extra leading index column is fine and is dropped automatically).

Run: python lis_advertising.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lis_core import (mse_loss, mse_grad, lis_gd_canonical, bgd_baseline,
                       add_bias, ols_closed_form)


def load_data(path="Advertising.csv"):
    df = pd.read_csv(path)
    # normalize column names, drop any unnamed index column
    df.columns = [c.strip() for c in df.columns]
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=drop_cols)
    cols_lower = {c.lower(): c for c in df.columns}
    feat_cols = [cols_lower[c] for c in ["tv", "radio", "newspaper"]]
    target_col = cols_lower["sales"]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def main():
    X_raw, y = load_data()
    print(f"Loaded Advertising data: n={X_raw.shape[0]}, d={X_raw.shape[1]}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42)

    mu, sigma = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    X_train_s = (X_train_raw - mu) / sigma
    X_test_s = (X_test_raw - mu) / sigma

    X_train = add_bias(X_train_s)
    X_test = add_bias(X_test_s)
    theta0 = np.zeros(X_train.shape[1])

    theta_star = ols_closed_form(X_train, y_train)
    tol = 1e-8

    # M_min scan: smallest M whose LIS-GD parameters are within tol of OLS
    M_min = None
    for M in range(5, 500):
        theta_M, _ = lis_gd_canonical(X_train, y_train, theta0, M)
        if np.max(np.abs(theta_M - theta_star)) <= tol:
            M_min = M
            break
    print(f"Candidate M (d_beta <= 1e-8) = {M_min}  (paper: 43)")

    # BGD baseline: run until convergence, report iterations + test MSE
    theta_bgd = theta0.copy()
    eta = 0.5  # stable constant LR on standardized design (lambda_max ~2.7)
    prev_loss = mse_loss(theta_bgd, X_train, y_train)
    for it in range(1, 20000):
        theta_bgd = theta_bgd - eta * mse_grad(theta_bgd, X_train, y_train)
        loss = mse_loss(theta_bgd, X_train, y_train)
        if abs(prev_loss - loss) < 1e-12:
            break
        prev_loss = loss
    test_mse_bgd = mse_loss(theta_bgd, X_test, y_test)
    print(f"BGD converged in {it} iterations, test MSE = {test_mse_bgd:.3f}"
          f"  (paper: 71 iters, 2.908)")

    # LIS-GD at M=49 (paper's headline single-profile comparison)
    theta_lis, _ = lis_gd_canonical(X_train, y_train, theta0, M=49)
    test_mse_lis = mse_loss(theta_lis, X_test, y_test)
    print(f"LIS-GD (M=49) test MSE = {test_mse_lis:.3f}  (paper: 2.908)")

    # Divergence sweep up to M=2000
    diverged = []
    for M in [50, 100, 500, 1000, 2000]:
        theta_M, _ = lis_gd_canonical(X_train, y_train, theta0, M)
        if not np.all(np.isfinite(theta_M)):
            diverged.append(M)
    print(f"Divergence observed at M in {diverged if diverged else 'none (matches paper up to M=2000)'}")


if __name__ == "__main__":
    main()
