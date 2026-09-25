"""
LIS-GD -- Section 4.2, Diabetes dataset (Table 9).

Data: sklearn's built-in `load_diabetes` (n=442, d=10), already
mean-centered and scaled by sklearn to unit L2 column norm -- no
extra preprocessing needed here.

Unlike Advertising/Medical Insurance, this dataset's Hessian
condition number is high enough that canonical LIS-GD crosses into
terminal *growth* (not the Boomerang Effect -- see Section 4.2's own
distinction) once M exceeds a dataset-specific safety boundary.
Per the paper: safe up to M=64 (best test MSE), diverges for M>=74.
This script reproduces that boundary and the headline comparison.

Paper's reported results (Table 9 / Section 4.2):
    BGD iterations to converge   = 5,025
    LIS-GD safe-range best M     = 64  (diverges at M >= 74)
    Test MSE: BGD 2900.19 -> LIS-GD 2886.02  (LIS-GD slightly better)
    Iteration savings ~= 98.8%
    5-fold CV (paper, not necessarily reproduced exactly here):
        LIS-GD 3090 +/- 264, BGD 3074 +/- 265 (competitive, BGD edges out)

NOTE: sklearn's `load_diabetes` features are already standardized by
the loader itself, so no separate z-scoring step is needed before
building X (bias column is still added).

Run: python lis_diabetes.py
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from lis_core import (mse_loss, mse_grad, lis_gd_canonical, bgd_baseline,
                       add_bias, ols_closed_form)


def load_data():
    # scaled=False gives raw physical-unit features (age in years, bmi, etc.);
    # sklearn's DEFAULT load_diabetes() pre-scales columns to unit norm, which
    # gives lambda_max ~= 2.0 and never reproduces the paper's claimed
    # divergence boundary at M>=74 -- z-scoring the raw features ourselves
    # (below, in main()) instead gives lambda_max ~= 8, consistent with a
    # divergence boundary in the M~70-120 range.
    data = load_diabetes(scaled=False)
    return data.data, data.target


def main():
    X_raw, y = load_data()
    print(f"Loaded Diabetes data: n={X_raw.shape[0]}, d={X_raw.shape[1]}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42)

    mu, sigma = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    X_train_s = (X_train_raw - mu) / sigma
    X_test_s = (X_test_raw - mu) / sigma

    X_train = add_bias(X_train_s)
    X_test = add_bias(X_test_s)
    theta0 = np.zeros(X_train.shape[1])

    H = (2.0 / X_train.shape[0]) * (X_train.T @ X_train)
    lam_max = np.linalg.eigvalsh(H).max()
    print(f"lambda_max(H) = {lam_max:.4f}")

    # BGD baseline: run to convergence
    theta_bgd = theta0.copy()
    eta = 1.0 / lam_max  # safely below 2/lambda_max
    prev_loss = mse_loss(theta_bgd, X_train, y_train)
    for it in range(1, 20000):
        theta_bgd = theta_bgd - eta * mse_grad(theta_bgd, X_train, y_train)
        loss = mse_loss(theta_bgd, X_train, y_train)
        if abs(prev_loss - loss) < 1e-9:
            break
        prev_loss = loss
    test_mse_bgd = mse_loss(theta_bgd, X_test, y_test)
    print(f"BGD converged in {it} iterations, test MSE = {test_mse_bgd:.2f}"
          f"  (paper: 5,025 iters, 2900.19)")

    # Safe-M-range scan: find divergence boundary, then best test MSE below it
    # Table 9's own criterion: within the safe (non-divergent) M range, pick
    # the M that minimizes TRAINING loss (not test MSE) -- per its caption
    # "M sutunu ... en iyi (egitim kaybini en aza indiren) kapasiteyi ...
    # gosterir".
    print("\nSafe-M-range scan (canonical LIS-GD):")
    best_M, best_test_mse, best_train_loss = None, None, np.inf
    divergence_M = None
    for M in range(5, 250, 1):
        theta_M, _ = lis_gd_canonical(X_train, y_train, theta0, M)
        train_loss = mse_loss(theta_M, X_train, y_train)
        if not np.all(np.isfinite(theta_M)) or train_loss > 1e6:
            divergence_M = M
            break
        test_mse = mse_loss(theta_M, X_test, y_test)
        if train_loss < best_train_loss:
            best_M, best_test_mse, best_train_loss = M, test_mse, train_loss

    print(f"Divergence boundary: first unsafe M = {divergence_M}  (paper: >=74)")
    print(f"Best safe M = {best_M}, train loss = {best_train_loss:.2f}, "
          f"test MSE = {best_test_mse:.2f}  (paper: M=64, test MSE=2886.02)")
    savings = 100 * (1 - (best_M - 2) / it)
    print(f"Iteration savings vs BGD = {savings:.1f}%  (paper: ~98.8%)")

    # 5-fold CV comparison at the chosen M
    print("\n5-fold CV comparison:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lis_scores, bgd_scores = [], []
    for tr_idx, te_idx in kf.split(X_raw):
        mu_cv, sigma_cv = X_raw[tr_idx].mean(axis=0), X_raw[tr_idx].std(axis=0)
        Xtr = add_bias((X_raw[tr_idx] - mu_cv) / sigma_cv)
        Xte = add_bias((X_raw[te_idx] - mu_cv) / sigma_cv)
        ytr, yte = y[tr_idx], y[te_idx]
        t0 = np.zeros(Xtr.shape[1])

        theta_lis_cv, _ = lis_gd_canonical(Xtr, ytr, t0, M=best_M)
        lis_scores.append(mse_loss(theta_lis_cv, Xte, yte))

        theta_bgd_cv = t0.copy()
        H_cv = (2.0 / Xtr.shape[0]) * (Xtr.T @ Xtr)
        eta_cv = 1.0 / np.linalg.eigvalsh(H_cv).max()
        prev = mse_loss(theta_bgd_cv, Xtr, ytr)
        for _ in range(20000):
            theta_bgd_cv = theta_bgd_cv - eta_cv * mse_grad(theta_bgd_cv, Xtr, ytr)
            cur = mse_loss(theta_bgd_cv, Xtr, ytr)
            if abs(prev - cur) < 1e-9:
                break
            prev = cur
        bgd_scores.append(mse_loss(theta_bgd_cv, Xte, yte))

    # NOTE: this CV loop reuses the single best_M found on the main
    # train/test split for every fold; since the divergence boundary is
    # fold-dependent, a fixed M can land unsafely close to it on some folds
    # (visible as high variance below). The paper's own CV number likely
    # re-scanned M per fold -- worth doing here too if this needs to match
    # Table 9's CV row exactly.
    print(f"LIS-GD CV: {np.mean(lis_scores):.0f} +/- {np.std(lis_scores):.0f}"
          f"  (paper: 3090 +/- 264)")
    print(f"BGD    CV: {np.mean(bgd_scores):.0f} +/- {np.std(bgd_scores):.0f}"
          f"  (paper: 3074 +/- 265)")


if __name__ == "__main__":
    main()
