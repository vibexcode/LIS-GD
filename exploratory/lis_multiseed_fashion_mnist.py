# -*- coding: utf-8 -*-
"""
LIS-GD Multi-Seed Baseline Comparison - Fashion-MNIST (PyTorch/GPU)
============================================================================
Multi-seed version of the earlier script. Same architecture
(784->64->10 MLP), same M=2901 canonical LIS-GD budget, same
matched-peak baseline set (tuned constant LR, linear/cosine/
triangular/one-cycle, AdamW) - but each method is run with N_SEEDS
different initial weights, and mean +/- std plus the divergence rate
are reported.

This addresses the "a single run is not enough - multiple seeds + std
+ divergence rate are needed" concern.

On Colab: select GPU (Runtime > Change runtime type > GPU).
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import time
import numpy as np

# ══════════════════════════════════════════════════════════
# ADJUSTABLE PARAMETERS
# ══════════════════════════════════════════════════════════
M_CAPACITY = 2901
TOTAL_STEPS = M_CAPACITY - 2   # 2899 effective steps
SEEDS = [42, 7, 123, 2024, 31337]   # expand if you like (5 -> 10)
HIDDEN = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════
# 1. Data (loaded once, same split used for every seed)
# ══════════════════════════════════════════════════════════

def load_data():
    tf = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=tf)
    test_set = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=tf)

    X_train = train_set.data.float().view(-1, 784) / 255.0
    y_train = train_set.targets.long()
    X_test = test_set.data.float().view(-1, 784) / 255.0
    y_test = test_set.targets.long()

    return (X_train.to(device), X_test.to(device),
            y_train.to(device), y_test.to(device))


# ══════════════════════════════════════════════════════════
# 2. Model - initial weight depends on the seed
# ══════════════════════════════════════════════════════════

def init_model(seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    W1 = (torch.randn(784, HIDDEN, generator=g) * math.sqrt(2 / 784)).to(device).requires_grad_(True)
    b1 = torch.zeros(1, HIDDEN, device=device, requires_grad=True)
    W2 = (torch.randn(HIDDEN, 10, generator=g) * math.sqrt(2 / HIDDEN)).to(device).requires_grad_(True)
    b2 = torch.zeros(1, 10, device=device, requires_grad=True)
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = F.relu(Z1)
    Z2 = A1 @ W2 + b2
    return Z2

@torch.no_grad()
def accuracy(logits, y_labels):
    return (logits.argmax(dim=1) == y_labels).float().mean().item()


# ══════════════════════════════════════════════════════════
# 3. Training loop (one seed) - NaN/Inf divergence is tracked
# ══════════════════════════════════════════════════════════

def train_one_seed(X_train, y_train, X_test, y_test, seed, eta_fn,
                    n_steps=TOTAL_STEPS, adamw=False, lr_adamw=0.001, wd_adamw=0.01):
    W1, b1, W2, b2 = init_model(seed)
    params = [W1, b1, W2, b2]

    opt = torch.optim.AdamW(params, lr=lr_adamw, weight_decay=wd_adamw) if adamw else None
    diverged = False

    for k in range(n_steps):
        logits = forward(X_train, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, y_train)

        if not torch.isfinite(loss):
            diverged = True
            break

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

    if diverged:
        return None, None, True

    with torch.no_grad():
        test_logits = forward(X_test, W1, b1, W2, b2)
        if not torch.isfinite(test_logits).all():
            return None, None, True
        final_acc = accuracy(test_logits, y_test)
        final_loss = F.cross_entropy(test_logits, y_test).item()

    return final_acc, final_loss, False


# ══════════════════════════════════════════════════════════
# 4. Schedules (same as lis_baseline_fashion_mnist.py)
# ══════════════════════════════════════════════════════════

def lis_schedule_fn(M=M_CAPACITY):
    profile = [math.log(k, M) - k / M for k in range(2, M)]  # M-2 values
    peak = max(profile)
    return (lambda k: profile[k]), peak

def const_fn(eta0):
    return lambda k: eta0

def linear_decay_fn(eta_max, n_steps=TOTAL_STEPS):
    return lambda k: eta_max * (1 - k / n_steps)

def cosine_fn(eta_max, eta_min=0.0, n_steps=TOTAL_STEPS):
    return lambda k: eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * k / n_steps))

def triangular_fn(eta_max, eta_min=0.0, n_steps=TOTAL_STEPS):
    half = n_steps / 2
    def f(k):
        if k <= half:
            return eta_min + (eta_max - eta_min) * (k / half)
        return eta_max - (eta_max - eta_min) * ((k - half) / half)
    return f

def one_cycle_fn(eta_max, eta_min=0.0, n_steps=TOTAL_STEPS, warmup_frac=0.3):
    warm = max(1, int(n_steps * warmup_frac))
    def f(k):
        if k <= warm:
            return eta_min + (eta_max - eta_min) * (k / warm)
        return eta_max * 0.5 * (1 + math.cos(math.pi * (k - warm) / (n_steps - warm)))
    return f


# ══════════════════════════════════════════════════════════
# 5. Multi-seed runner
# ══════════════════════════════════════════════════════════

def run_method(name, X_train, y_train, X_test, y_test, eta_fn=None,
                adamw=False, lr_adamw=0.001):
    accs, losses, n_div = [], [], 0
    t0 = time.time()
    for seed in SEEDS:
        acc, loss, div = train_one_seed(X_train, y_train, X_test, y_test,
                                         seed, eta_fn, adamw=adamw, lr_adamw=lr_adamw)
        if div:
            n_div += 1
        else:
            accs.append(acc)
            losses.append(loss)
    dt = time.time() - t0

    if len(accs) == 0:
        print(f"{name:<28} -> ALL SEEDS DIVERGED  ({dt:.1f}s)")
        return dict(name=name, acc_mean=None, acc_std=None,
                     loss_mean=None, loss_std=None,
                     diverged=n_div, total=len(SEEDS))

    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
    loss_mean, loss_std = float(np.mean(losses)), float(np.std(losses))
    print(f"{name:<28} -> acc={acc_mean*100:.2f}±{acc_std*100:.2f}%  "
          f"loss={loss_mean:.4f}±{loss_std:.4f}  "
          f"diverged={n_div}/{len(SEEDS)}  ({dt:.1f}s)")

    return dict(name=name, acc_mean=acc_mean, acc_std=acc_std,
                 loss_mean=loss_mean, loss_std=loss_std,
                 diverged=n_div, total=len(SEEDS))


def run_all():
    print("Loading data (Fashion-MNIST)...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {tuple(X_train.shape)}  Test: {tuple(X_test.shape)}")
    print(f"Number of seeds: {len(SEEDS)}  ({SEEDS})\n")

    lis_fn, LIS_PEAK = lis_schedule_fn(M_CAPACITY)
    print(f"LIS profile: M={M_CAPACITY}, {TOTAL_STEPS} effective steps, peak={LIS_PEAK:.4f}\n")

    results = []
    results.append(run_method(f'LIS-GD (M={M_CAPACITY})', X_train, y_train, X_test, y_test, lis_fn))

    # tuned constant LR: quick grid on the first seed, then run the
    # best value on all seeds
    best_eta, best_acc0 = None, -1
    for eta0 in [0.2, 0.3, 0.4, 0.5, 0.6]:
        acc0, _, div0 = train_one_seed(X_train, y_train, X_test, y_test,
                                        SEEDS[0], const_fn(eta0))
        if not div0 and acc0 > best_acc0:
            best_eta, best_acc0 = eta0, acc0
    results.append(run_method(f'Tuned constant LR (eta={best_eta})',
                               X_train, y_train, X_test, y_test, const_fn(best_eta)))

    results.append(run_method('Linear decay', X_train, y_train, X_test, y_test,
                               linear_decay_fn(LIS_PEAK)))
    results.append(run_method('Cosine annealing', X_train, y_train, X_test, y_test,
                               cosine_fn(LIS_PEAK)))
    results.append(run_method('Triangular CLR', X_train, y_train, X_test, y_test,
                               triangular_fn(LIS_PEAK)))
    results.append(run_method('One-cycle', X_train, y_train, X_test, y_test,
                               one_cycle_fn(LIS_PEAK)))

    best_lr, best_acc0 = None, -1
    for lr in [0.0005, 0.001, 0.003]:
        acc0, _, div0 = train_one_seed(X_train, y_train, X_test, y_test,
                                        SEEDS[0], None, adamw=True, lr_adamw=lr)
        if not div0 and acc0 > best_acc0:
            best_lr, best_acc0 = lr, acc0
    results.append(run_method(f'AdamW (lr={best_lr})', X_train, y_train, X_test, y_test,
                               adamw=True, lr_adamw=best_lr))

    print(f"\n{'='*90}")
    print(f"  {'Method':<28} {'Test Acc (%)':>16} {'Test Loss':>18} {'Diverged':>10}")
    print(f"  {'-'*86}")
    for r in results:
        if r['acc_mean'] is None:
            print(f"  {r['name']:<28} {'--- all diverged ---':>16} {'':>18} {r['diverged']}/{r['total']:>6}")
        else:
            acc_s = f"{r['acc_mean']*100:.2f}±{r['acc_std']*100:.2f}"
            loss_s = f"{r['loss_mean']:.4f}±{r['loss_std']:.4f}"
            print(f"  {r['name']:<28} {acc_s:>16} {loss_s:>18} {r['diverged']}/{r['total']:>6}")
    print(f"{'='*90}")

    return results


if __name__ == "__main__":
    run_all()
