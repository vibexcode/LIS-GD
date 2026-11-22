# -*- coding: utf-8 -*-
"""
Fair baseline comparison for LIS-GD on MNIST (PyTorch/GPU version)
=======================================================================
Identical protocol to the NumPy version (same data loading/split, same
architecture, same seed, same 499-step full-batch budget), but
computation is done with torch tensors on GPU - a large speedup on
Colab with a T4 GPU (full-batch 56000x784 forward/backward is fast on
GPU, slower on NumPy/CPU).

Data is loaded via sklearn.fetch_openml (NOT torchvision.datasets.
MNIST) - because the reference result (Section 4.2, 94.89%) uses this
exact pipeline; a different loader gives a different train/test split
and the numbers become incomparable. Only the compute layer has been
moved to torch; the data/protocol is preserved exactly.

On Colab: select Runtime > Change runtime type > GPU (T4).
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

RNG_SEED = 42
HIDDEN = 64
TOTAL_STEPS = 499
LIS_PEAK = 0.5451

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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

    X_train_t = torch.from_numpy(X_train).to(device)
    X_test_t  = torch.from_numpy(X_test).to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    y_test_t  = torch.from_numpy(y_test).long().to(device)

    y_train_oh = F.one_hot(y_train_t, num_classes=10).float()
    y_test_oh  = F.one_hot(y_test_t, num_classes=10).float()

    return X_train_t, X_test_t, y_train_oh, y_test_oh, y_train_t, y_test_t


# ══════════════════════════════════════════════════════════
# 2. Model - same architecture (784 -> 64 -> 10), raw parameters
#    (not nn.Module + optimizer, so that every learning-rate schedule
#    can be applied by hand using plain tensor parameters)
# ══════════════════════════════════════════════════════════

def init_model():
    g = torch.Generator(device='cpu').manual_seed(RNG_SEED)
    W1 = (torch.randn(784, HIDDEN, generator=g) * math.sqrt(2 / 784)).to(device).requires_grad_(True)
    b1 = torch.zeros(1, HIDDEN, device=device, requires_grad=True)
    W2 = (torch.randn(HIDDEN, 10, generator=g) * math.sqrt(2 / HIDDEN)).to(device).requires_grad_(True)
    b2 = torch.zeros(1, 10, device=device, requires_grad=True)
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = F.relu(Z1)
    Z2 = A1 @ W2 + b2
    return Z2  # logits; softmax+CE applied together

@torch.no_grad()
def accuracy(logits, y_labels):
    preds = logits.argmax(dim=1)
    return (preds == y_labels).float().mean().item()


# ══════════════════════════════════════════════════════════
# 3. Generic training loop
# ══════════════════════════════════════════════════════════

def train_with_schedule(X_train, y_train_oh, y_train_labels,
                         X_test, y_test_oh, y_test_labels,
                         eta_fn, n_steps=TOTAL_STEPS, verbose=False,
                         adamw=False, lr_adamw=0.001, wd_adamw=0.01):
    W1, b1, W2, b2 = init_model()
    params = [W1, b1, W2, b2]

    opt = None
    if adamw:
        opt = torch.optim.AdamW(params, lr=lr_adamw, weight_decay=wd_adamw)

    for k in range(n_steps):
        logits = forward(X_train, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, y_train_labels)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()
            eta = eta_fn(k)
            with torch.no_grad():
                for p in params:
                    p -= eta * p.grad

        if verbose and k % 100 == 0:
            with torch.no_grad():
                test_logits = forward(X_test, W1, b1, W2, b2)
                print(f"    k={k:3d}  test_acc={accuracy(test_logits, y_test_labels):.4f}")

    with torch.no_grad():
        test_logits = forward(X_test, W1, b1, W2, b2)
        final_acc = accuracy(test_logits, y_test_labels)
        final_loss = F.cross_entropy(test_logits, y_test_labels).item()

    return final_acc, final_loss


# ══════════════════════════════════════════════════════════
# 4. Schedules - same definitions as the NumPy version
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
    X_train, X_test, y_train_oh, y_test_oh, y_train_lab, y_test_lab = load_data()
    print(f"Train: {tuple(X_train.shape)}  Test: {tuple(X_test.shape)}\n")

    results = {}

    def go(name, eta_fn=None, adamw=False, lr_adamw=0.001):
        t0 = time.time()
        acc, loss = train_with_schedule(
            X_train, y_train_oh, y_train_lab, X_test, y_test_oh, y_test_lab,
            eta_fn, adamw=adamw, lr_adamw=lr_adamw
        )
        dt = time.time() - t0
        print(f"{name:<28} -> acc={acc:.4f}  loss={loss:.4f}  ({dt:.1f}s)")
        return acc, loss, dt

    results['LIS-GD (M=500)'] = go('LIS-GD (M=500)', lis_schedule_fn(500))

    best = (None, -1, None)
    for eta0 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        acc_i, loss_i, _ = go(f'  [grid] const eta={eta0:.2f}', const_fn(eta0))
        if acc_i > best[1]:
            best = (eta0, acc_i, loss_i)
    results[f'Tuned constant LR (eta={best[0]})'] = (best[1], best[2], None)
    print(f"Tuned constant LR (eta={best[0]})        -> acc={best[1]:.4f}  loss={best[2]:.4f}")

    results['Linear decay']     = go('Linear decay', linear_decay_fn())
    results['Cosine annealing'] = go('Cosine annealing', cosine_fn())
    results['Triangular CLR']   = go('Triangular CLR', triangular_fn())
    results['One-cycle']        = go('One-cycle', one_cycle_fn())

    best_adamw = (None, -1, None)
    for lr in [0.0005, 0.001, 0.003, 0.01]:
        acc_i, loss_i, _ = go(f'  [grid] AdamW lr={lr:.4f}', adamw=True, lr_adamw=lr)
        if acc_i > best_adamw[1]:
            best_adamw = (lr, acc_i, loss_i)
    results[f'AdamW (lr={best_adamw[0]})'] = (best_adamw[1], best_adamw[2], None)
    print(f"AdamW (lr={best_adamw[0]})            -> acc={best_adamw[1]:.4f}  loss={best_adamw[2]:.4f}")

    print(f"\n{'='*66}")
    print(f"  {'Method':<28} {'Test Acc':>10} {'Test Loss':>12}")
    print(f"  {'-'*62}")
    for name, val in results.items():
        acc, loss = val[0], val[1]
        print(f"  {name:<28} {acc:>10.4f} {loss:>12.4f}")
    print(f"{'='*66}")

    return results


if __name__ == "__main__":
    run_all()
