"""
Canonical LIS-GD core implementation.

Reproduces the algorithm exactly as defined in the manuscript
"LIS-GD: Sonlu Ufuklu Gradyan Optimizasyonu" (Bolum 2.3, Eq. 3):

    I_extra(k; M) = log_M(k) - k/M = ln(k)/ln(M) - k/M      (Eq. 1)

    theta_{n+1} = theta_n - I_extra(k; M) * grad L(theta_n)
    k = n + 2,  n = 0, ..., M-2                              (Eq. 3)

so the schedule has exactly M-2 effective updates; the terminal
coefficient I_extra(M; M) = 0 and is skipped by construction (k only
ever reaches M-1 in the loop, since n goes up to M-2 => k = n+2 up to M).

Loss convention (matches Section 3.5's H = (2/m) X^T X):
    L(theta) = (1/m) * ||X theta - y||^2   (mean squared error, NOT halved)
    grad L(theta) = (2/m) * X^T (X theta - y)

This file has no dependency on any specific dataset; each
lis_<dataset>.py script imports `lis_gd_canonical` and `bgd_baseline`
from here and supplies its own data-loading / preprocessing.
"""

import numpy as np


def i_extra(k, M):
    """LIS coefficient, Eq. (1). k and M are integers, k in {1,...,M}."""
    k = np.asarray(k, dtype=float)
    return np.log(k) / np.log(M) - k / M


def mse_loss(theta, X, y):
    """L(theta) = (1/m) ||X theta - y||^2, X already includes bias column."""
    resid = X @ theta - y
    m = X.shape[0]
    return float((resid @ resid) / m)


def mse_grad(theta, X, y):
    """grad L(theta) = (2/m) X^T (X theta - y)."""
    m = X.shape[0]
    resid = X @ theta - y
    return (2.0 / m) * (X.T @ resid)


def lis_gd_canonical(X, y, theta0, M, loss_fn=mse_loss, grad_fn=mse_grad,
                      track_history=False):
    """
    Canonical (single-profile) LIS-GD, full-batch, Eq. (3).

    Runs exactly M-2 effective updates (n = 0, ..., M-2), matching the
    paper's convention that the terminal step k=M carries I_extra=0 and
    is not applied.

    Returns
    -------
    theta : final parameter vector
    history : list of (k, I_extra(k,M), loss) if track_history else None
    """
    theta = theta0.copy().astype(float)
    history = [] if track_history else None
    for n in range(0, M - 1):  # n = 0, ..., M-2  -> M-1 values, k = 2..M
        k = n + 2
        if k > M - 1:
            # k == M gives I_extra(M;M) = 0 exactly -> no-op, skip (per Eq. 3 note)
            break
        coef = i_extra(k, M)
        g = grad_fn(theta, X, y)
        theta = theta - coef * g
        if track_history:
            history.append((k, coef, loss_fn(theta, X, y)))
    return theta, history


def bgd_baseline(X, y, theta0, eta, n_iter, loss_fn=mse_loss, grad_fn=mse_grad,
                  track_history=False):
    """Untuned/tuned constant-learning-rate batch gradient descent baseline."""
    theta = theta0.copy().astype(float)
    history = [] if track_history else None
    for _ in range(n_iter):
        g = grad_fn(theta, X, y)
        theta = theta - eta * g
        if track_history:
            history.append(loss_fn(theta, X, y))
    return theta, history


def add_bias(X):
    """Prepend a column of ones for the intercept term."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def ols_closed_form(X, y):
    """theta* via normal equations, X already includes bias column."""
    return np.linalg.lstsq(X, y, rcond=None)[0]
