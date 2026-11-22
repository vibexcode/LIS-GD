"""
LIS-GD -- Section 3.2 controlled synthetic linear regression (Ek A.1).

Reproduces the manuscript's own headline validation experiment:
    seed = 42
    x ~ U(0, 2), 100 samples
    y = 4 + 3x + eps,  eps ~ N(0, 1)
    theta0 (fixed for ALL methods) = (0.4967, -0.1383)

Verifies against the manuscript's reported quantities:
    L_OLS            = 0.806584564
    lambda_min        = 0.3394
    lambda_max        = 4.1301
    eta_crit = 2/lambda_max ~= 0.4843
    M_min(eps=1e-8)   = 83   (first M whose LIS-GD terminal loss is
                              within 1e-8 of L_OLS)
    eta = 0.48  -> converges in 692 steps (stable, below eta_crit)
    eta = 0.49  -> diverges (above eta_crit)

Run: python lis_synthetic_regression.py
"""

import numpy as np
from lis_core import (i_extra, mse_loss, mse_grad, lis_gd_canonical,
                       bgd_baseline, add_bias, ols_closed_form)


def generate_data(seed=42, n=100):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 2, size=(n, 1))
    eps = rng.normal(0, 1, size=(n, 1))
    y = 4 + 3 * x + eps
    return x, y.ravel()


def main():
    x, y = generate_data(seed=42, n=100)
    X = add_bias(x)  # [1, x]
    m = X.shape[0]

    # OLS reference
    theta_star = ols_closed_form(X, y)
    L_OLS = mse_loss(theta_star, X, y)
    print(f"OLS solution theta* = {theta_star}")
    print(f"L_OLS = {L_OLS:.9f}  (paper: 0.806584564)")

    # Hessian H = (2/m) X^T X, its eigenvalues
    H = (2.0 / m) * (X.T @ X)
    eigvals = np.linalg.eigvalsh(H)
    lam_min, lam_max = eigvals.min(), eigvals.max()
    eta_crit = 2.0 / lam_max
    print(f"lambda_min = {lam_min:.4f}  (paper: 0.3394)")
    print(f"lambda_max = {lam_max:.4f}  (paper: 4.1301)")
    print(f"eta_crit = 2/lambda_max = {eta_crit:.4f}  (paper: ~0.4843)")

    # Fixed start point used for ALL methods (paper's theta0^GD)
    theta0 = np.array([0.4967, -0.1383])

    # --- M_min(eps=1e-8) scan ---
    tol = 1e-8
    M_min = None
    for M in range(5, 300):
        theta_M, _ = lis_gd_canonical(X, y, theta0, M)
        loss_M = mse_loss(theta_M, X, y)
        if abs(loss_M - L_OLS) <= tol:
            M_min = M
            break
    print(f"M_min(eps=1e-8) = {M_min}  (paper: 83)")

    # --- Table 1 spot-checks: M = 50, 83, 100, 164 ---
    for M in [50, 83, 100, 164]:
        theta_M, hist = lis_gd_canonical(X, y, theta0, M, track_history=True)
        loss_M = mse_loss(theta_M, X, y)
        I_max = max(c for (_, c, _) in hist)
        print(f"M={M:4d}: final loss = {loss_M:.6f}, "
              f"|loss - L_OLS| = {abs(loss_M - L_OLS):.3e}, I_max = {I_max:.5f}")

    # --- eta = 0.48 (stable, ~692 steps) vs eta = 0.49 (diverges) ---
    theta_48, hist_48 = bgd_baseline(X, y, theta0, eta=0.48, n_iter=692,
                                      track_history=True)
    print(f"BGD eta=0.48, 692 iters: final loss = {hist_48[-1]:.6f} "
          f"(should be close to L_OLS = {L_OLS:.6f})")

    theta_49, hist_49 = bgd_baseline(X, y, theta0, eta=0.49, n_iter=50,
                                      track_history=True)
    print(f"BGD eta=0.49, 50 iters: final loss = {hist_49[-1]:.3e} "
          f"(should diverge / blow up)")


if __name__ == "__main__":
    main()
