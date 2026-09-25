"""
LIS-GD -- Section 4.2, Energy Efficiency dataset (Table 9).

Data: UCI ENB2012 (Tsanas & Xifara, 2012) -- building shape/glazing
features X1-X8 predicting heating load Y1, n=768, d=8.

Same safe-M-range methodology as lis_diabetes.py: LIS-GD crosses into
terminal growth past a dataset-specific M, so M is chosen as the
training-loss-minimizing point within the safe (non-divergent) range.

Paper's reported results (Table 9 / Section 4.2):
    BGD iterations to converge = 6,965
    LIS-GD safe M              = 100  (diverges at M >= 119)
    Test MSE: BGD 9.153 -> LIS-GD 9.343  (LIS-GD slightly worse, ~2%)
    Iteration savings ~= 98.6%

Requires: ENB2012_data.csv in the same directory (X1..X8,Y1,Y2 columns).

Run: python lis_energy_efficiency.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lis_core import mse_loss, mse_grad, lis_gd_canonical, add_bias


def load_data(path="ENB2012_data.csv", target="Y1"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    feat_cols = [c for c in df.columns if c.upper().startswith("X")]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    return X, y


def main():
    X_raw, y = load_data(target="Y1")  # Y1 = heating load, per the manuscript
    print(f"Loaded Energy Efficiency data: n={X_raw.shape[0]}, d={X_raw.shape[1]}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42)

    mu, sigma = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train = add_bias((X_train_raw - mu) / sigma)
    X_test = add_bias((X_test_raw - mu) / sigma)
    theta0 = np.zeros(X_train.shape[1])

    H = (2.0 / X_train.shape[0]) * (X_train.T @ X_train)
    lam_max = np.linalg.eigvalsh(H).max()
    print(f"lambda_max(H) = {lam_max:.4f}")

    # BGD baseline
    theta_bgd = theta0.copy()
    eta = 1.0 / lam_max
    prev_loss = mse_loss(theta_bgd, X_train, y_train)
    for it in range(1, 20000):
        theta_bgd = theta_bgd - eta * mse_grad(theta_bgd, X_train, y_train)
        loss = mse_loss(theta_bgd, X_train, y_train)
        if abs(prev_loss - loss) < 1e-9:
            break
        prev_loss = loss
    test_mse_bgd = mse_loss(theta_bgd, X_test, y_test)
    print(f"BGD converged in {it} iterations, test MSE = {test_mse_bgd:.3f}"
          f"  (paper: 6,965 iters, 9.153)")

    # Safe-M-range scan, training-loss-minimizing selection (Table 9 criterion)
    print("\nSafe-M-range scan (canonical LIS-GD):")
    best_M, best_test_mse, best_train_loss = None, None, np.inf
    divergence_M = None
    for M in range(5, 400):
        theta_M, _ = lis_gd_canonical(X_train, y_train, theta0, M)
        train_loss = mse_loss(theta_M, X_train, y_train)
        if not np.all(np.isfinite(theta_M)) or train_loss > 1e6:
            divergence_M = M
            break
        test_mse = mse_loss(theta_M, X_test, y_test)
        if train_loss < best_train_loss:
            best_M, best_test_mse, best_train_loss = M, test_mse, train_loss

    print(f"Divergence boundary: first unsafe M = {divergence_M}  (paper: >=119)")
    print(f"Best safe M = {best_M}, train loss = {best_train_loss:.3f}, "
          f"test MSE = {best_test_mse:.3f}  (paper: M=100, test MSE=9.343)")
    savings = 100 * (1 - (best_M - 2) / it)
    print(f"Iteration savings vs BGD = {savings:.1f}%  (paper: ~98.6%)")


if __name__ == "__main__":
    main()
