# -*- coding: utf-8 -*-
"""
LIS-GD: Minimum-capacity scan over correlated multi-dimensional
synthetic regression problems.

Same methodology as Section 3.5.1 of the paper: instead of a step-by-
step GD simulation, the closed-form spectral product from Equations
(10)-(11) is used to directly compute the loss/parameter difference
for each candidate M, and M is scanned. This keeps results independent
of float64 simulation noise and iteration-level rounding error.

Difference from Section 3.5.1: that section scans a single 2D
(lambda_min, lambda_max) problem via a scale factor. Here we build a
synthetic regression problem with d>2 features that have a controlled
correlation structure (AR(1)-type: Sigma_ij = rho^|i-j|), and sweep
the correlation parameter rho, growing the Hessian condition number.
M_min(eps) is searched for at each rho level; if none is found, it is
reported as "Undefined".

Requires only numpy (pip install numpy).
"""

import numpy as np

RNG_SEED = 42
EPS_TOLERANCE = 1e-8

# ══════════════════════════════════════════════════════════
# 1. LIS profile - Equation (1)
# ══════════════════════════════════════════════════════════

def i_extra(k, M):
    """I_extra(k; M) = log_M(k) - k/M,  k=1,...,M. Negative for k=1;
    in practice k=2,...,M-1 is used (Section 2.2)."""
    k = np.asarray(k, dtype=np.float64)
    return np.log(k) / np.log(M) - k / M


# ══════════════════════════════════════════════════════════
# 2. Closed-form spectral product P_M(lambda) - Equation (16)
# ══════════════════════════════════════════════════════════

def P_M(lam, M):
    """|P_M(lambda)| = prod_{k=2}^{M-1} |1 - I_extra(k;M) * lambda|
    Summed in log-space (to avoid overflow/underflow), then
    exponentiated back. lam can be a scalar or an array."""
    ks = np.arange(2, M)  # k = 2, ..., M-1
    coeffs = i_extra(ks, M)  # shape (M-2,)
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    with np.errstate(divide="ignore"):
        log_abs = np.log(np.abs(1.0 - np.outer(lam, coeffs)) + 1e-300)
    log_sum = np.clip(log_abs.sum(axis=1), -700, 700)
    return np.exp(log_sum)  # shape (len(lam),)


# ══════════════════════════════════════════════════════════
# 3. M_min(eps) search - same logic as Section 3.5.1
# ══════════════════════════════════════════════════════════

def find_m_min(eigvals, c0, eps=EPS_TOLERANCE, M_grid=None):
    """For a given set of eigenvalues and initial modal projections
    (c0, same order), scans the closed-form total loss difference
    from Equation (45) over an M grid and returns the first M that
    satisfies dL(M) <= eps. Returns None if not found ("Undefined")."""
    if M_grid is None:
        M_grid = np.unique(np.round(
            np.geomspace(10, 2_000_000, 400)
        ).astype(np.int64))
        M_grid = M_grid[M_grid >= 6]

    for M in M_grid:
        with np.errstate(over="ignore", invalid="ignore"):
            pm = P_M(eigvals, int(M))
            dL = np.sum(0.5 * (c0 ** 2) * (pm ** 2) * eigvals)
        # E_{i,0} = 0.5*lambda_i*c_{i,0}^2 (Equation 45); c0 is the
        # projection of e0 onto the eigenbasis
        if np.isfinite(dL) and dL <= eps:
            return int(M), float(dL)
    return None, None


# ══════════════════════════════════════════════════════════
# 4. Correlated synthetic regression problem setup
# ══════════════════════════════════════════════════════════

def make_correlated_problem(d, rho, n_samples=300, noise_std=1.0, seed=42):
    """d-dimensional (excluding bias) synthetic regression problem
    with AR(1)-type correlation structure: Sigma_ij = rho^|i-j|.
    Hessian H = (2/n) X^T X (same definition as Section 3.5); a bias
    column (x0=1) is appended to X."""
    rng = np.random.default_rng(seed)

    idx = np.arange(d)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(d))

    Z = rng.standard_normal((n_samples, d))
    Xf = Z @ L.T  # correlated features, N(0, Sigma)

    w_true = rng.standard_normal(d)
    y = Xf @ w_true + 4.0 + noise_std * rng.standard_normal(n_samples)

    X = np.hstack([np.ones((n_samples, 1)), Xf])  # bias column added

    H = (2.0 / n_samples) * (X.T @ X)
    eigvals, eigvecs = np.linalg.eigh(H)  # ascending order

    theta_star = np.linalg.lstsq(X, y, rcond=None)[0]  # OLS
    theta0 = np.zeros_like(theta_star)                 # zero init

    e0 = theta0 - theta_star
    c0 = eigvecs.T @ e0  # projection onto eigenbasis

    return eigvals, c0, theta_star


# ══════════════════════════════════════════════════════════
# 5. Main sweep: M_min as a function of correlation level
# ══════════════════════════════════════════════════════════

def run_sweep(d=6, rhos=(0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.97, 0.99),
              n_samples=300, eps=EPS_TOLERANCE, seed=RNG_SEED):
    print(f"{'rho':>6} {'lambda_min':>12} {'lambda_max':>12} "
          f"{'kappa':>10} {'M_min(eps)':>12} {'dL':>12}")
    print("-" * 68)

    rows = []
    for rho in rhos:
        eigvals, c0, theta_star = make_correlated_problem(
            d=d, rho=rho, n_samples=n_samples, seed=seed
        )
        lam_min = eigvals.min()
        lam_max = eigvals.max()
        kappa = lam_max / lam_min

        m_min, dL = find_m_min(eigvals, c0, eps=eps)

        m_str = f"{m_min}" if m_min is not None else "Undefined"
        dl_str = f"{dL:.2e}" if dL is not None else "-"

        print(f"{rho:>6.2f} {lam_min:>12.4f} {lam_max:>12.4f} "
              f"{kappa:>10.2f} {m_str:>12} {dl_str:>12}")

        rows.append(dict(rho=rho, lambda_min=lam_min, lambda_max=lam_max,
                          kappa=kappa, M_min=m_min, dL=dL))
    return rows


# ══════════════════════════════════════════════════════════
# 6. (Optional) Cross-check with direct GD - a few points
# ══════════════════════════════════════════════════════════

def verify_with_direct_gd(d, rho, M, n_samples=300, seed=RNG_SEED):
    """Verifies the closed-form result by actually running step-by-
    step GD. Recommended for only a few (rho, M) pairs (slow)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(d)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(d))
    Z = rng.standard_normal((n_samples, d))
    Xf = Z @ L.T
    w_true = rng.standard_normal(d)
    y = Xf @ w_true + 4.0 + rng.standard_normal(n_samples)
    X = np.hstack([np.ones((n_samples, 1)), Xf])

    theta_star = np.linalg.lstsq(X, y, rcond=None)[0]
    theta = np.zeros_like(theta_star)

    for k in range(2, M):  # k = 2, ..., M-1 (Equation 3)
        coeff = np.log(k) / np.log(M) - k / M
        grad = (2.0 / n_samples) * X.T @ (X @ theta - y)
        theta = theta - coeff * grad

    dL_direct = np.mean((X @ theta - y) ** 2) - np.mean((X @ theta_star - y) ** 2)
    return theta, dL_direct


if __name__ == "__main__":
    print("LIS-GD: Multi-dimensional synthetic correlation sweep")
    print("=" * 68)
    rows = run_sweep(d=6, n_samples=300, eps=1e-8)

    print("\nCross-check (closed-form vs direct GD, first successful M):")
    for r in rows:
        if r["M_min"] is not None:
            _, dL_direct = verify_with_direct_gd(6, r["rho"], r["M_min"])
            print(f"  rho={r['rho']:.2f}  M={r['M_min']:6d}  "
                  f"closed-form dL={r['dL']:.3e}  "
                  f"direct-GD dL={dL_direct:.3e}")
