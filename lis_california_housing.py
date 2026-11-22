# -*- coding: utf-8 -*-
"""
California Housing - canonical LIS-GD verification (5-fold CV)
==================================================================
Independently reproduces the California Housing result reported in
the source notes (BGD 506 iter vs LIS-GD M=130, 74% savings, test MSE
0.555654 vs 0.555847). Methodology: standardized features + bias, BGD
(eta=0.1, tolerance 1e-9) as reference, then the minimum M for
LIS-GD that reaches BGD's training-loss level is scanned. Followed by
a 5-fold CV to check generalizability.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

SEED = 42

def i_extra(k, M):
    return np.log(k) / np.log(M) - k / M

def run_bgd(X, y, eta=0.1, tol=1e-9, max_iter=5000):
    n, d = X.shape
    theta = np.zeros(d)
    prev_loss = np.inf
    for it in range(1, max_iter + 1):
        grad = (2.0 / n) * X.T @ (X @ theta - y)
        theta = theta - eta * grad
        loss = np.mean((X @ theta - y) ** 2)
        if abs(prev_loss - loss) < tol:
            return theta, it, loss
        prev_loss = loss
    return theta, max_iter, loss

def run_lis_gd(X, y, M):
    n, d = X.shape
    theta = np.zeros(d)
    for k in range(2, M):
        grad = (2.0 / n) * X.T @ (X @ theta - y)
        theta = theta - i_extra(k, M) * grad
    return theta

def find_matching_M(X, y, target_train_loss, M_grid=range(10, 2000)):
    n = len(y)
    for M in M_grid:
        theta = run_lis_gd(X, y, M)
        loss = np.mean((X @ theta - y) ** 2)
        if loss <= target_train_loss * 1.0001:  # match BGD's level
            return M, loss
    return None, None


def main():
    print("Loading California Housing data...")
    data = fetch_california_housing()
    X_raw, y = data.data.astype(np.float64), data.target.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED)

    scaler = StandardScaler().fit(X_train)
    X_train_s = np.hstack([np.ones((len(X_train), 1)), scaler.transform(X_train)])
    X_test_s = np.hstack([np.ones((len(X_test), 1)), scaler.transform(X_test)])

    print("\n--- Main train/test split ---")
    theta_bgd, n_iter_bgd, train_loss_bgd = run_bgd(X_train_s, y_train)
    test_mse_bgd = np.mean((X_test_s @ theta_bgd - y_test) ** 2)
    print(f"BGD: {n_iter_bgd} iterations, train_loss={train_loss_bgd:.6f}, "
          f"test_MSE={test_mse_bgd:.6f}")

    M_match, lis_train_loss = find_matching_M(X_train_s, y_train, train_loss_bgd)
    if M_match is None:
        print("LIS-GD: BGD's level was not reached in the scanned range.")
        return
    theta_lis = run_lis_gd(X_train_s, y_train, M_match)
    test_mse_lis = np.mean((X_test_s @ theta_lis - y_test) ** 2)
    savings = 100 * (1 - (M_match - 2) / n_iter_bgd)
    print(f"LIS-GD: M={M_match} ({M_match-2} effective steps), "
          f"train_loss={lis_train_loss:.6f}, test_MSE={test_mse_lis:.6f}")
    print(f"Iteration savings: {savings:.1f}%")

    print("\n--- 5-fold Cross-Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    bgd_scores, lis_scores = [], []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_s), 1):
        Xtr, Xval = X_train_s[tr_idx], X_train_s[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        theta_b, _, tl_b = run_bgd(Xtr, ytr)
        mse_b = np.mean((Xval @ theta_b - yval) ** 2)

        M_f, _ = find_matching_M(Xtr, ytr, tl_b)
        theta_l = run_lis_gd(Xtr, ytr, M_f) if M_f else None
        mse_l = np.mean((Xval @ theta_l - yval) ** 2) if theta_l is not None else np.nan

        bgd_scores.append(mse_b)
        lis_scores.append(mse_l)
        print(f"  Fold {fold} — BGD: {mse_b:.6f} | LIS-GD (M={M_f}): {mse_l:.6f}")

    print(f"\nBGD CV mean: {np.mean(bgd_scores):.6f} ± {np.std(bgd_scores):.6f}")
    print(f"LIS CV mean: {np.nanmean(lis_scores):.6f} ± {np.nanstd(lis_scores):.6f}")


if __name__ == "__main__":
    main()
