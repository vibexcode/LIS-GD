# -*- coding: utf-8 -*-
"""
Fair baseline comparison for LIS-GD on Fashion-MNIST (PyTorch/GPU)
=======================================================================
Same logic as the MNIST script, applied to Fashion-MNIST. Data is
loaded via torchvision.datasets.FashionMNIST (NOTE: the paper's
original Fashion-MNIST result comes from a different codebase; the
purpose of this script is not to reproduce that exact absolute number,
but to compare LIS-GD against the other schedules under identical,
fair conditions - so do not directly mix the absolute numbers here
with the original Section 4.2 table).

Architecture: 784 -> 64 -> 10 MLP (same as the Fashion-MNIST
experiment). Budget: LIS-GD uses M=31 (as in the paper), the others
use the same step count (30 effective steps).

On Colab: select GPU (Runtime > Change runtime type > GPU).
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import time

RNG_SEED = 42
HIDDEN = 64
M_LIS = 31
TOTAL_STEPS = M_LIS - 1     # 30 effective steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════
# 1. Data
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
# 2. Model
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
    return Z2

@torch.no_grad()
def accuracy(logits, y_labels):
    return (logits.argmax(dim=1) == y_labels).float().mean().item()


# ══════════════════════════════════════════════════════════
# 3. Training loop
# ══════════════════════════════════════════════════════════

def train_with_schedule(X_train, y_train, X_test, y_test,
                         eta_fn, n_steps=TOTAL_STEPS,
                         adamw=False, lr_adamw=0.001, wd_adamw=0.01):
    W1, b1, W2, b2 = init_model()
    params = [W1, b1, W2, b2]

    opt = torch.optim.AdamW(params, lr=lr_adamw, weight_decay=wd_adamw) if adamw else None

    for k in range(n_steps):
        logits = forward(X_train, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, y_train)

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

    with torch.no_grad():
        test_logits = forward(X_test, W1, b1, W2, b2)
        final_acc = accuracy(test_logits, y_test)
        final_loss = F.cross_entropy(test_logits, y_test).item()

    return final_acc, final_loss


# ══════════════════════════════════════════════════════════
# 4. Schedules
# ══════════════════════════════════════════════════════════

def lis_schedule_fn(M=M_LIS):
    profile = [math.log(k, M) - k / M for k in range(2, M + 1)]
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
# 5. Main run
# ══════════════════════════════════════════════════════════

def run_all():
    print("Loading data (Fashion-MNIST)...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {tuple(X_train.shape)}  Test: {tuple(X_test.shape)}\n")

    lis_fn, LIS_PEAK = lis_schedule_fn(M_LIS)
    print(f"LIS profile: M={M_LIS}, {TOTAL_STEPS} steps, peak={LIS_PEAK:.4f}\n")

    results = {}

    def go(name, eta_fn=None, adamw=False, lr_adamw=0.001):
        t0 = time.time()
        acc, loss = train_with_schedule(X_train, y_train, X_test, y_test,
                                         eta_fn, adamw=adamw, lr_adamw=lr_adamw)
        dt = time.time() - t0
        print(f"{name:<28} -> acc={acc:.4f}  loss={loss:.4f}  ({dt:.1f}s)")
        return acc, loss

    results['LIS-GD (M=31)'] = go('LIS-GD (M=31)', lis_fn)

    best = (None, -1, None)
    for eta0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        acc_i, loss_i = go(f'  [grid] const eta={eta0:.2f}', const_fn(eta0))
        if acc_i > best[1]:
            best = (eta0, acc_i, loss_i)
    results[f'Tuned constant LR (eta={best[0]})'] = (best[1], best[2])
    print(f"Tuned constant LR (eta={best[0]})        -> acc={best[1]:.4f}  loss={best[2]:.4f}")

    results['Linear decay']     = go('Linear decay', linear_decay_fn(LIS_PEAK))
    results['Cosine annealing'] = go('Cosine annealing', cosine_fn(LIS_PEAK))
    results['Triangular CLR']   = go('Triangular CLR', triangular_fn(LIS_PEAK))
    results['One-cycle']        = go('One-cycle', one_cycle_fn(LIS_PEAK))

    best_adamw = (None, -1, None)
    for lr in [0.001, 0.003, 0.01, 0.03]:
        acc_i, loss_i = go(f'  [grid] AdamW lr={lr:.4f}', adamw=True, lr_adamw=lr)
        if acc_i > best_adamw[1]:
            best_adamw = (lr, acc_i, loss_i)
    results[f'AdamW (lr={best_adamw[0]})'] = (best_adamw[1], best_adamw[2])
    print(f"AdamW (lr={best_adamw[0]})            -> acc={best_adamw[1]:.4f}  loss={best_adamw[2]:.4f}")

    print(f"\n{'='*66}")
    print(f"  {'Method':<28} {'Test Acc':>10} {'Test Loss':>12}")
    print(f"  {'-'*62}")
    for name, (acc, loss) in results.items():
        print(f"  {name:<28} {acc:>10.4f} {loss:>12.4f}")
    print(f"{'='*66}")

    return results


if __name__ == "__main__":
    run_all()
