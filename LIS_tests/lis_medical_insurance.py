"""
LIS-GD -- Section 4.1, Medical Insurance dataset (Table 8).

Data: classic "Medical Cost Personal Datasets" insurance.csv
(age, sex, bmi, children, smoker, region -> charges),
n = 1338. One-hot encoding of sex/smoker/region (drop-first) gives
d = 8 numeric features + intercept, matching the paper's stated d=8.

Same protocol note as lis_advertising.py: plain linear regression,
full-batch GD, canonical LIS-GD, target d_beta <= 1e-8 tolerance to
the OLS solution; features standardized (fit on train only) for
numerical stability -- raw "charges"-scale features would give a
huge lambda_max and diverge under any LIS-GD/BGD schedule, which
would be inconsistent with the paper's own "no divergence up to
M=5000" claim for this dataset.

Paper's reported results (Table 8 / Section 4.1):
    Candidate M (tolerance-reaching)  = 774
    Single-run M for headline comparison = 10,002 (very large -- this
        dataset's Hessian is worse-conditioned than Advertising's, so
        it needs a much larger M to fully converge)
    BGD reaches test MSE = 33,596,916 in 216 iterations
    LIS-GD (M=86, per Section 4.1's follow-up paragraph) reaches the
        same 33,596,916 test MSE
    No divergence observed for any M <= 5000 tested

NOTE: as with Advertising, the manuscript does not state its exact
train/test split for the reported test-MSE figures; this script uses
an 80/20 split with random_state=42 as the default. If your own
reproduction used a different split, adjust `test_size`/`random_state`.

Requires: insurance.csv in the same directory (age,sex,bmi,children,
smoker,region,charges columns).

Run: python lis_medical_insurance.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lis_core import (mse_loss, mse_grad, lis_gd_canonical, bgd_baseline,
                       add_bias, ols_closed_form)


def load_data(path="insurance.csv"):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
    y = df.pop("charges").to_numpy(dtype=float)
    X = df.to_numpy(dtype=float)
    return X, y, list(df.columns)


def main():
    X_raw, y, feat_names = load_data()
    print(f"Loaded Medical Insurance data: n={X_raw.shape[0]}, d={X_raw.shape[1]}")
    print(f"Features: {feat_names}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42)

    mu, sigma = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    sigma[sigma == 0] = 1.0  # guard against degenerate binary columns
    X_train_s = (X_train_raw - mu) / sigma
    X_test_s = (X_test_raw - mu) / sigma

    X_train = add_bias(X_train_s)
    X_test = add_bias(X_test_s)
    theta0 = np.zeros(X_train.shape[1])

    H = (2.0 / X_train.shape[0]) * (X_train.T @ X_train)
    lam_max = np.linalg.eigvalsh(H).max()
    print(f"lambda_max(H) = {lam_max:.4f}  (should be small, O(1-10), for stability)")

    theta_star = ols_closed_form(X_train, y_train)
    tol = 1e-8

    M_min = None
    for M in range(5, 2000):
        theta_M, _ = lis_gd_canonical(X_train, y_train, theta0, M)
        if np.max(np.abs(theta_M - theta_star)) <= tol:
            M_min = M
            break
    print(f"Candidate M (d_beta <= 1e-8) = {M_min}  (paper: 774)")

    # BGD baseline
    theta_bgd = theta0.copy()
    eta = 0.5
    prev_loss = mse_loss(theta_bgd, X_train, y_train)
    for it in range(1, 20000):
        theta_bgd = theta_bgd - eta * mse_grad(theta_bgd, X_train, y_train)
        loss = mse_loss(theta_bgd, X_train, y_train)
        if abs(prev_loss - loss) < 1e-6:
            break
        prev_loss = loss
    test_mse_bgd = mse_loss(theta_bgd, X_test, y_test)
    print(f"BGD converged in {it} iterations, test MSE = {test_mse_bgd:,.2f}"
          f"  (paper: 216 iters, 33,596,916)")

    # LIS-GD at M=86 (paper's headline single-profile comparison, Section 4.1 text)
    theta_lis, _ = lis_gd_canonical(X_train, y_train, theta0, M=86)
    test_mse_lis = mse_loss(theta_lis, X_test, y_test)
    print(f"LIS-GD (M=86) test MSE = {test_mse_lis:,.2f}  (paper: 33,596,916)")

    diverged = []
    for M in [100, 500, 1000, 2000, 5000]:
        theta_M, _ = lis_gd_canonical(X_train, y_train, theta0, M)
        if not np.all(np.isfinite(theta_M)):
            diverged.append(M)
    print(f"Divergence observed at M in {diverged if diverged else 'none (matches paper up to M=5000)'}")


if __name__ == "__main__":
    main()
