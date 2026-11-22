"""
Table 3 (Advertising, Medical Insurance, California Housing): 10 independent
train/test splits, mean +/- std, for BGD iterations to reach d_beta <= 1e-8
at several learning rates, LIS-GD M_min(1e-8), and train/test MSE at the
converged (~OLS) point.

Data files (same directory): Advertising.csv, insurance.csv.
California Housing is read from california_housing.csv (column "target")
if present, otherwise downloaded through scikit-learn.

Note: the d_beta check below uses the maximum absolute coordinate
difference |theta - theta_OLS|_inf <= tol.

Run: python lis_regression_10seed.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lis_core import mse_grad, lis_gd_canonical, add_bias, ols_closed_form


def bgd_iters_to_tol(X, y, theta0, theta_star, eta, tol=1e-8, max_iter=1_000_000):
    theta = theta0.copy().astype(float)
    for it in range(1, max_iter + 1):
        theta = theta - eta * mse_grad(theta, X, y)
        if np.max(np.abs(theta - theta_star)) <= tol:
            return it
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e8:
            return None  # diverged
    return None  # did not converge within max_iter


def lis_M_to_tol(X, y, theta0, theta_star, tol=1e-8, M_max=5000):
    for M in range(5, M_max):
        theta_M, _ = lis_gd_canonical(X, y, theta0, M)
        if not np.all(np.isfinite(theta_M)):
            return None
        if np.max(np.abs(theta_M - theta_star)) <= tol:
            return M
    return None


def load_advertising():
    df = pd.read_csv("Advertising.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
    cols = {c.lower(): c for c in df.columns}
    X = df[[cols["tv"], cols["radio"], cols["newspaper"]]].to_numpy(dtype=float)
    y = df[cols["sales"]].to_numpy(dtype=float)
    return X, y


def load_insurance():
    df = pd.read_csv("insurance.csv")
    df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
    y = df.pop("charges").to_numpy(dtype=float)
    X = df.to_numpy(dtype=float)
    return X, y


def load_california():
    if os.path.exists("california_housing.csv"):
        df = pd.read_csv("california_housing.csv")
        y = df.pop("target").to_numpy(dtype=float)
        X = df.to_numpy(dtype=float)
        return X, y
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    return d.data.astype(float), d.target.astype(float)


def analyze_dataset(name, loader, seeds, M_max, bgd_max_iter):
    rows = []
    for seed in seeds:
        X_raw, y_raw = loader()
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=seed)
        scX = StandardScaler()
        X_train = add_bias(scX.fit_transform(X_train_raw))
        X_test = add_bias(scX.transform(X_test_raw))

        theta0 = np.zeros(X_train.shape[1])
        theta_star = ols_closed_form(X_train, y_train)

        H = (2.0 / X_train.shape[0]) * (X_train.T @ X_train)
        lam_max = np.linalg.eigvalsh(H).max()
        opt_eta = 1.0 / lam_max

        row = {"seed": seed, "lambda_max": lam_max, "opt_eta": opt_eta}
        for label, eta in [("0.1", 0.1), ("0.3", 0.3), ("0.5", 0.5), ("opt", opt_eta)]:
            it = bgd_iters_to_tol(X_train, y_train, theta0, theta_star, eta,
                                   max_iter=bgd_max_iter)
            row[f"BGD@{label}"] = it
        M_min = lis_M_to_tol(X_train, y_train, theta0, theta_star, M_max=M_max)
        row["LIS_M"] = M_min

        train_mse = np.mean((X_train @ theta_star - y_train) ** 2)
        test_mse = np.mean((X_test @ theta_star - y_test) ** 2)
        row["train_mse"] = train_mse
        row["test_mse"] = test_mse
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df, name):
    print("=" * 70)
    print(name)
    print("=" * 70)
    print(df.to_string(index=False))
    print("\nSummary (mean +/- std, over seeds where reached):")
    for col in ["lambda_max", "opt_eta", "BGD@0.1", "BGD@0.3", "BGD@0.5", "BGD@opt", "LIS_M", "train_mse", "test_mse"]:
        vals = df[col].dropna()
        n_diverged = df[col].isna().sum()
        if len(vals) > 0:
            print(f"  {col}: {vals.mean():.6g} +/- {vals.std():.6g}  (n_diverged/never={n_diverged})")
        else:
            print(f"  {col}: all diverged/never reached")
    print()


if __name__ == "__main__":
    seeds = list(range(10))

    df_adv = analyze_dataset("Advertising", load_advertising, seeds, M_max=2000, bgd_max_iter=500000)
    summarize(df_adv, "ADVERTISING")

    df_ins = analyze_dataset("Medical Insurance", load_insurance, seeds, M_max=3000, bgd_max_iter=500000)
    summarize(df_ins, "MEDICAL INSURANCE")

    df_ch = analyze_dataset("California Housing", load_california, seeds, M_max=3000, bgd_max_iter=500000)
    summarize(df_ch, "CALIFORNIA HOUSING")
