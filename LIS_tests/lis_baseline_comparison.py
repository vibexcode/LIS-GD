# -*- coding: utf-8 -*-
"""
Fair baseline comparison for LIS-GD on MNIST (NumPy/CPU)
============================================================
Compares canonical LIS-GD to schedules with a matched budget (tuned
constant, linear decay, cosine annealing, triangular CLR, one-cycle)
and to AdamW, all starting from identical initial weights and using
full-batch gradient descent on a 784->64->10 MLP.

CPU/NumPy version (a PyTorch/GPU version is also provided). Data is
loaded via sklearn.datasets.fetch_openml('mnist_784').
"""

import numpy as np
import math
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time

RNG_SEED = 42
HIDDEN = 64
TOTAL_STEPS = 499          # same budget as LIS-GD (M=500)
LIS_PEAK = 0.5451          # observed peak coefficient in the LIS-GD result

# ══════════════════════════════════════════════════════════
# 1. Data - same loading/split as the reference experiment
# ══════════════════════════════════════════════════════════

def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED
    )

    def one_hot(yy, n_classes=10):
        Y = np.zeros((len(yy), n_classes))
        Y[np.arange(len(yy)), yy] = 1
        return Y

    return X_train, X_test, one_hot(y_train), one_hot(y_test), y_train, y_test


# ══════════════════════════════════════════════════════════
# 2. Model functions - identical to the reference experiment
# ══════════════════════════════════════════════════════════

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(y_hat, y_true):
    return -np.mean(np.sum(y_true * np.log(y_hat + 1e-15), axis=1))

def accuracy(y_hat, y):
    return np.mean(np.argmax(y_hat, axis=1) == y)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward(X, y_true, Z1, A1, A2, W2):
    m = len(X)
    dZ2 = A2 - y_true
    dW2 = (1 / m) * A1.T @ dZ2
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = (dZ2 @ W2.T) * relu_deriv(Z1)
    dW1 = (1 / m) * X.T @ dZ1
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

def init_model():
    """Every method starts from IDENTICAL initial weights, same seed."""
    rng = np.random.RandomState(RNG_SEED)
    W1 = rng.randn(784, HIDDEN) * np.sqrt(2 / 784)
    b1 = np.zeros((1, HIDDEN))
    W2 = rng.randn(HIDDEN, 10) * np.sqrt(2 / HIDDEN)
    b2 = np.zeros((1, 10))
    return W1, b1, W2, b2


# ══════════════════════════════════════════════════════════
# 3. Generic training loop - takes an "eta(k)" function
# ══════════════════════════════════════════════════════════

def train_with_schedule(X_train, y_train_oh, X_test, y_test_oh, y_test,
                         eta_fn, n_steps=TOTAL_STEPS, verbose=False,
                         adamw=False, lr_adamw=0.001, wd_adamw=0.01):
    """eta_fn(k) -> the learning rate at step k (k=0,...,n_steps-1).
    If adamw=True, an AdamW update (lr_adamw, wd_adamw) is used
    instead of eta_fn."""
    W1, b1, W2, b2 = init_model()

    if adamw:
        mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
        mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
        mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
        mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

    for k in range(n_steps):
        Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X_train, y_train_oh, Z1, A1, A2, W2)

        if adamw:
            t = k + 1
            for (p, g, m_, v_) in [
                (W1, dW1, mW1, vW1), (b1, db1, mb1, vb1),
                (W2, dW2, mW2, vW2), (b2, db2, mb2, vb2)
            ]:
                m_ *= beta1; m_ += (1 - beta1) * g
                v_ *= beta2; v_ += (1 - beta2) * (g ** 2)
                m_hat = m_ / (1 - beta1 ** t)
                v_hat = v_ / (1 - beta2 ** t)
                p -= lr_adamw * (m_hat / (np.sqrt(v_hat) + eps) + wd_adamw * p)
        else:
            eta = eta_fn(k)
            W1 -= eta * dW1; b1 -= eta * db1
            W2 -= eta * dW2; b2 -= eta * db2

        if verbose and k % 100 == 0:
            _, _, _, tp = forward(X_test, W1, b1, W2, b2)
            print(f"    k={k:3d}  test_acc={accuracy(tp, y_test):.4f}")

    _, _, _, test_probs = forward(X_test, W1, b1, W2, b2)
    final_acc = accuracy(test_probs, y_test)
    final_loss = cross_entropy(test_probs, y_test_oh)
    return final_acc, final_loss


# ══════════════════════════════════════════════════════════
# 4. Schedules
# ══════════════════════════════════════════════════════════

def lis_schedule_fn(M=500):
    profile = [math.log(k, M) - k / M for k in range(2, M + 1)]  # 499 values
    return lambda k: profile[k]

def const_fn(eta0):
    return lambda k: eta0

def linear_decay_fn(eta_max=LIS_PEAK, n_steps=TOTAL_STEPS):
    return lambda k: eta_max * (1 - k / n_steps)

def cosine_fn(eta_max=LIS_PEAK, eta_min=0.0, n_steps=TOTAL_STEPS):
    return lambda k: eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * k / n_steps))

def triangular_fn(eta_max=LIS_PEAK, eta_min=0.0, n_steps=TOTAL_STEPS):
    half = n_steps / 2
    def f(k):
        if k <= half:
            return eta_min + (eta_max - eta_min) * (k / half)
        return eta_max - (eta_max - eta_min) * ((k - half) / half)
    return f

def one_cycle_fn(eta_max=LIS_PEAK, eta_min=0.0, n_steps=TOTAL_STEPS, warmup_frac=0.3):
    warm = int(n_steps * warmup_frac)
    def f(k):
        if k <= warm:
            return eta_min + (eta_max - eta_min) * (k / warm)
        return eta_max * 0.5 * (1 + math.cos(math.pi * (k - warm) / (n_steps - warm)))
    return f


# ══════════════════════════════════════════════════════════
# 5. Main run
# ══════════════════════════════════════════════════════════

def run_all():
    print("Loading data...")
    X_train, X_test, y_train_oh, y_test_oh, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}\n")

    results = {}

    # --- 1) LIS-GD reference ---
    t0 = time.time()
    acc, loss = train_with_schedule(X_train, y_train_oh, X_test, y_test_oh, y_test,
                                     lis_schedule_fn(500))
    results['LIS-GD (M=500)'] = (acc, loss, time.time() - t0)
    print(f"LIS-GD (M=500)              -> acc={acc:.4f}  loss={loss:.4f}")

    # --- 2) Tuned constant LR (small grid) ---
    best = (None, -1, None)
    for eta0 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        acc_i, loss_i = train_with_schedule(X_train, y_train_oh, X_test, y_test_oh,
                                             y_test, const_fn(eta0))
        print(f"  [grid] const eta={eta0:.2f}  -> acc={acc_i:.4f}")
        if acc_i > best[1]:
            best = (eta0, acc_i, loss_i)
    results[f'Tuned constant LR (eta={best[0]})'] = (best[1], best[2], None)
    print(f"Tuned constant LR (eta={best[0]})        -> acc={best[1]:.4f}  loss={best[2]:.4f}")

    # --- 3) Linear decay ---
    t0 = time.time()
    acc, loss = train_with_schedule(X_train, y_train_oh, X_test, y_test_oh, y_test,
                                     linear_decay_fn())
    results['Linear decay'] = (acc, loss, time.time() - t0)
    print(f"Linear decay                -> acc={acc:.4f}  loss={loss:.4f}")

    # --- 4) Cosine annealing ---
    t0 = time.time()
    acc, loss = train_with_schedule(X_train, y_train_oh, X_test, y_test_oh, y_test,
                                     cosine_fn())
    results['Cosine annealing'] = (acc, loss, time.time() - t0)
    print(f"Cosine annealing             -> acc={acc:.4f}  loss={loss:.4f}")

    # --- 5) Triangular CLR ---
    t0 = time.time()
    acc, loss = train_with_schedule(X_train, y_train_oh, X_test, y_test_oh, y_test,
                                     triangular_fn())
    results['Triangular CLR'] = (acc, loss, time.time() - t0)
    print(f"Triangular CLR               -> acc={acc:.4f}  loss={loss:.4f}")

    # --- 6) One-cycle ---
    t0 = time.time()
    acc, loss = train_with_schedule(X_train, y_train_oh, X_test, y_test_oh, y_test,
                                     one_cycle_fn())
    results['One-cycle'] = (acc, loss, time.time() - t0)
    print(f"One-cycle                   -> acc={acc:.4f}  loss={loss:.4f}")

    # --- 7) AdamW (small lr grid) ---
    best_adamw = (None, -1, None)
    for lr in [0.0005, 0.001, 0.003, 0.01]:
        acc_i, loss_i = train_with_schedule(X_train, y_train_oh, X_test, y_test_oh,
                                             y_test, None, adamw=True, lr_adamw=lr)
        print(f"  [grid] AdamW lr={lr:.4f}  -> acc={acc_i:.4f}")
        if acc_i > best_adamw[1]:
            best_adamw = (lr, acc_i, loss_i)
    results[f'AdamW (lr={best_adamw[0]})'] = (best_adamw[1], best_adamw[2], None)
    print(f"AdamW (lr={best_adamw[0]})            -> acc={best_adamw[1]:.4f}  loss={best_adamw[2]:.4f}")

    # --- Summary table ---
    print(f"\n{'='*66}")
    print(f"  {'Method':<28} {'Test Acc':>10} {'Test Loss':>12}")
    print(f"  {'-'*62}")
    for name, (acc, loss, _) in results.items():
        print(f"  {name:<28} {acc:>10.4f} {loss:>12.4f}")
    print(f"{'='*66}")

    return results


if __name__ == "__main__":
    run_all()
