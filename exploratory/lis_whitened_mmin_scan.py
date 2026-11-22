# -*- coding: utf-8 -*-
"""
M_min scan for canonical LIS-GD after whitening (PCA+whitening)
==========================================================================
For the House Prices / Superconductivity / Bike Sharing datasets, after
whitening the feature matrix (targeting H ~ 2I) to shrink the Hessian's
condition number, runs an M_min(eps) scan using ONLY canonical
(single-profile, no Cycle-LIS) LIS-GD.

Methodology is IDENTICAL to Section 3.5.1 of the paper: instead of a
step-by-step GD simulation, the closed-form spectral product
P_M(lambda) is used to compute the loss/parameter difference directly
for each candidate M, and M is scanned. This means the computation
finishes in seconds no matter how large n (sample count) is - only the
eigendecomposition of the d x d Hessian is needed (d = number of
features + bias), no GPU required.

DATA SOURCES: the load_* functions below first try an automatic
download via openml. If that fails, place the CSV by hand under
./data/ and update the DOSYA_ADI (file name) variable inside each
function (see the comments).
"""

import numpy as np

RNG_SEED = 42
EPS_TOLERANCE = 1e-8


# ══════════════════════════════════════════════════════════
# 1. LIS profile and closed-form spectral product
#    (same as lis_correlation_sweep.py, same as Equations 1/16)
# ══════════════════════════════════════════════════════════

def i_extra(k, M):
    k = np.asarray(k, dtype=np.float64)
    return np.log(k) / np.log(M) - k / M

def P_M(lam, M):
    ks = np.arange(2, M)
    coeffs = i_extra(ks, M)
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    with np.errstate(divide="ignore"):
        log_abs = np.log(np.abs(1.0 - np.outer(lam, coeffs)) + 1e-300)
    log_sum = np.clip(log_abs.sum(axis=1), -700, 700)
    return np.exp(log_sum)

def find_m_min(eigvals, c0, eps=EPS_TOLERANCE, M_grid=None):
    if M_grid is None:
        M_grid = np.unique(np.round(np.geomspace(6, 2_000_000, 500)).astype(np.int64))
    for M in M_grid:
        with np.errstate(over="ignore", invalid="ignore"):
            pm = P_M(eigvals, int(M))
            dL = np.sum(0.5 * (c0 ** 2) * (pm ** 2) * eigvals)
        if np.isfinite(dL) and dL <= eps:
            return int(M), float(dL)
    return None, None


# ══════════════════════════════════════════════════════════
# 2. Whitening + M_min scan - one function, called for each
#    dataset
# ══════════════════════════════════════════════════════════

def whiten_and_scan(X, y, name, eps=EPS_TOLERANCE):
    """X: (n, d_raw) feature matrix (excluding bias). y: (n,) target.
    First centers+standardizes, then applies full PCA+whitening
    (targeting H ~ 2I, same definition as Section 5.6 of the paper),
    finally adds a bias column and runs the canonical M_min scan."""
    n, d = X.shape
    print(f"\n{'='*60}\n{name}  (n={n}, d_raw={d})\n{'='*60}")

    # Center + standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    y = y - y.mean()  # also center the target (bias handled separately)

    # Full PCA (SVD)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Rank-deficiency check: drop near-zero singular-value directions
    # (exactly linearly dependent / zero-variance components, e.g.
    # engineered features that are exact sums of others). These
    # directions carry no real information and, if kept, break the
    # bias-orthogonality of the whitened representation.
    tol = S.max() * max(n, d) * np.finfo(S.dtype).eps
    keep = S > tol
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        print(f"  [note] {n_dropped} near-zero singular value(s) detected "
              f"(rank deficiency) - these directions were dropped from PCA.")
    U, S = U[:, keep], S[keep]
    d_eff = keep.sum()

    # Whitening: Z = U * sqrt(n) (population-variance normalization,
    # same as Equation 63 of the paper): Z^T Z / n = I
    Z = U * np.sqrt(n)

    # Add bias column (unwhitened, constant term)
    Xb = np.hstack([np.ones((n, 1)), Z])

    H = (2.0 / n) * (Xb.T @ Xb)
    eigvals, eigvecs = np.linalg.eigh(H)
    lam_min, lam_max = eigvals.min(), eigvals.max()
    kappa = lam_max / lam_min
    print(f"  After whitening: lambda_min={lam_min:.6f}  "
          f"lambda_max={lam_max:.6f}  kappa={kappa:.4f}")

    theta_star = np.linalg.lstsq(Xb, y, rcond=None)[0]
    theta0 = np.zeros_like(theta_star)  # zero init
    c0 = eigvecs.T @ (theta0 - theta_star)

    m_min, dL = find_m_min(eigvals, c0, eps=eps)
    if m_min is not None:
        print(f"  --> Canonical LIS-GD M_min(eps={eps:.0e}) = {m_min}  "
              f"(effective steps = {m_min-2}, verified dL={dL:.3e})")
    else:
        print(f"  --> Canonical LIS-GD: no M_min found in the scanned range (Undefined)")

    return dict(name=name, n=n, d=d, lam_min=lam_min, lam_max=lam_max,
                kappa=kappa, m_min=m_min, dL=dL)


def raw_scan(X, y, name, eps=EPS_TOLERANCE):
    """Same M_min scan on the RAW Hessian (centering+standardization+
    bias only, no whitening). Used as a comparison reference."""
    n, d = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    y = y - y.mean()
    Xb = np.hstack([np.ones((n, 1)), X])

    H = (2.0 / n) * (Xb.T @ Xb)
    eigvals, eigvecs = np.linalg.eigh(H)
    lam_min, lam_max = eigvals.min(), eigvals.max()
    kappa = lam_max / lam_min
    print(f"  [RAW, no whitening] lambda_min={lam_min:.6f}  "
          f"lambda_max={lam_max:.6f}  kappa={kappa:.2f}")

    theta_star = np.linalg.lstsq(Xb, y, rcond=None)[0]
    theta0 = np.zeros_like(theta_star)
    c0 = eigvecs.T @ (theta0 - theta_star)

    m_min, dL = find_m_min(eigvals, c0, eps=eps)
    if m_min is not None:
        print(f"  --> [RAW] Canonical LIS-GD M_min(eps={eps:.0e}) = {m_min} "
              f"(effective steps = {m_min-2})")
    else:
        print(f"  --> [RAW] Canonical LIS-GD: Undefined (not found in the scanned range)")

    return dict(name=name + " (raw)", n=n, d=d, lam_min=lam_min, lam_max=lam_max,
                kappa=kappa, m_min=m_min, dL=dL)


# ══════════════════════════════════════════════════════════
# 3. Data loaders
# ══════════════════════════════════════════════════════════

def load_superconductivity():
    """UCI Superconductivty Data (21263 rows, 81 features +
    critical_temp). Tries openml first; otherwise update the CSV
    path."""
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="Superconductivty", version=1, as_frame=True)
        df = data.frame
        y = df["critical_temp"].values.astype(np.float64)
        X = df.drop(columns=["critical_temp"]).values.astype(np.float64)
        return X, y
    except Exception as e:
        print(f"  [warning] automatic download from openml failed: {e}")
        print("  Download manually: https://archive.ics.uci.edu/dataset/464/superconductivty+data")
        print("  and edit the line below to match your CSV path:")
        # import pandas as pd
        # df = pd.read_csv("./data/train.csv")
        # y = df["critical_temp"].values; X = df.drop(columns=["critical_temp"]).values
        raise

def load_bike_sharing():
    """UCI Bike Sharing (day.csv, 731 rows). May also be found on
    openml under the name 'Bike_Sharing_Demand'; otherwise update the
    CSV path."""
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True)
        df = data.frame
        # drop non-numeric / leakage columns (casual+registered=cnt, etc.)
        y = df["count"].values.astype(np.float64) if "count" in df.columns else df.iloc[:, -1].values.astype(np.float64)
        num_df = df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in ["count"] if c in df.columns], errors="ignore")
        X = num_df.values.astype(np.float64)
        return X, y
    except Exception as e:
        print(f"  [warning] automatic download from openml failed: {e}")
        print("  Download manually: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset")
        print("  (day.csv) and edit the CSV path.")
        raise

def load_house_prices():
    """Kaggle 'House Prices - Advanced Regression Techniques'
    (train.csv). Requires Kaggle authentication, so it CANNOT be
    downloaded automatically - place the file by hand at
    ./data/house_prices_train.csv. Alternatively you can use
    sklearn's California Housing dataset (see the commented-out
    line below)."""
    import os
    path = "./data/house_prices_train.csv"
    if not os.path.exists(path):
        print(f"  [warning] {path} not found.")
        print("  Download from Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques")
        print("  and place train.csv at this path. Alternative: use load_california_housing_as_proxy().")
        raise FileNotFoundError(path)
    import pandas as pd
    df = pd.read_csv(path)
    y = df["SalePrice"].values.astype(np.float64)
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["SalePrice", "Id"], errors="ignore")
    num_df = num_df.fillna(num_df.median())
    X = num_df.values.astype(np.float64)
    return X, y

def load_california_housing_as_proxy():
    """A readily accessible alternative (built into sklearn) if
    House Prices cannot be found."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    return data.data.astype(np.float64), data.target.astype(np.float64)


# ══════════════════════════════════════════════════════════
# 4. Main run
# ══════════════════════════════════════════════════════════

def run_all():
    results = []

    try:
        X, y = load_superconductivity()
        results.append(whiten_and_scan(X, y, "Superconductivity"))
    except Exception:
        print("  Superconductivity skipped.")

    try:
        X, y = load_bike_sharing()
        results.append(whiten_and_scan(X, y, "Bike Sharing"))
    except Exception:
        print("  Bike Sharing skipped.")

    try:
        X, y = load_house_prices()
        results.append(whiten_and_scan(X, y, "House Prices"))
    except Exception:
        print("  House Prices skipped (CSV not found) - trying California Housing instead...")
        try:
            X, y = load_california_housing_as_proxy()
            results.append(whiten_and_scan(X, y, "California Housing (proxy)"))
        except Exception as e2:
            print(f"  California Housing also failed: {e2}")

    print(f"\n{'='*70}")
    print(f"  {'Dataset':<26} {'n':>7} {'d':>5} {'kappa (whitened)':>17} {'M_min':>10}")
    print(f"  {'-'*66}")
    for r in results:
        m_str = str(r['m_min']) if r['m_min'] is not None else "Undefined"
        print(f"  {r['name']:<26} {r['n']:>7} {r['d']:>5} {r['kappa']:>17.4f} {m_str:>10}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    run_all()
