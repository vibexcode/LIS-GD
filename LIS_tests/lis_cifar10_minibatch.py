# -*- coding: utf-8 -*-
"""
LIS-GD (mini-batch, canonical) exploration on CIFAR-10
=====================================================================
A small CNN for CIFAR-10 classification. LIS-GD is used here in
mini-batch form (IDENTICAL to the "LIS-scheduled SGD" definition in
Section 2.3 of the paper): each mini-batch step uses the next k in the
profile, with NO coefficient ever repeated or held (this is NOT
Cycle-LIS or Spread-LIS). Total step count is M-2 (M = capacity).

This script also includes a Cycle-LIS mode (cycle_lis_coeffs) used for
an EXPLORATORY comparison only (not part of the paper) - repeating a
smaller, stable capacity M several times to reach the same total
budget as a larger single-pass run.

On Colab: select GPU (Runtime > Change runtime type > GPU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import time
import numpy as np

# ══════════════════════════════════════════════════════════
# ADJUSTABLE PARAMETERS
# ══════════════════════════════════════════════════════════
BATCH_SIZE = 128
SEEDS = [42, 7, 123]        # 3 seeds for cost reasons (expand if you like)

# For the Cycle-LIS exploration:
CYCLE_M = 1000              # small capacity observed to be stable
CYCLE_C = 3                 # number of repetitions (total ~ (M-2)*C steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════
# 1. Data
# ══════════════════════════════════════════════════════════

def load_data():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=tf)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=tf)
    return train_set, test_set


# ══════════════════════════════════════════════════════════
# 2. Small CNN
# ══════════════════════════════════════════════════════════

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 4x4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.net(x))


def make_model(seed):
    torch.manual_seed(seed)
    return SmallCNN().to(device)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction='sum').item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return correct / total, loss_sum / total


# ══════════════════════════════════════════════════════════
# 3. LIS profile (canonical, mini-batch - Section 2.3) and
#    Cycle-LIS coefficients (exploratory only)
# ══════════════════════════════════════════════════════════

def lis_profile(M):
    return [math.log(k, M) - k / M for k in range(2, M)]  # M-2 values

def cycle_lis_coeffs(M, C):
    """Cycle-LIS: repeat the M-capacity profile C times (total (M-2)*C)."""
    single = lis_profile(M)
    return single * C


# ══════════════════════════════════════════════════════════
# 4. Training loop (one seed)
# ══════════════════════════════════════════════════════════

def train_one(seed, train_set, test_loader, n_steps, mode,
              eta_const=None, lr_adamw=None, coeffs=None):
    """mode: 'cycle' | 'const' | 'adamw'
    coeffs: the precomputed coefficient array used in 'cycle' mode
    (length must equal n_steps)."""
    model = make_model(seed)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2, drop_last=True)

    if mode == 'cycle':
        profile = coeffs
        assert len(profile) == n_steps
    opt = torch.optim.AdamW(model.parameters(), lr=lr_adamw) if mode == 'adamw' else None

    step = 0
    diverged = False
    data_iter = iter(train_loader)
    while step < n_steps:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)

        model.train()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        if not torch.isfinite(loss):
            diverged = True
            break

        model.zero_grad()
        loss.backward()

        if opt is not None:
            opt.step()
        else:
            eta = profile[step] if mode in ('lis', 'cycle') else eta_const
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= eta * p.grad
        step += 1

    if diverged:
        return None, None, True

    acc, loss_val = evaluate(model, test_loader)
    if not np.isfinite(acc) or not np.isfinite(loss_val):
        return None, None, True
    return acc, loss_val, False


# ══════════════════════════════════════════════════════════
# 5. Multi-seed runner
# ══════════════════════════════════════════════════════════

def run_method(name, train_set, test_loader, n_steps, **kwargs):
    accs, losses, n_div = [], [], 0
    div_seeds, ok_seeds = [], []
    t0 = time.time()
    for seed in SEEDS:
        acc, loss, div = train_one(seed, train_set, test_loader, n_steps, **kwargs)
        if div:
            n_div += 1
            div_seeds.append(seed)
        else:
            accs.append(acc)
            losses.append(loss)
            ok_seeds.append(seed)
    dt = time.time() - t0
    print(f"  [detail] diverged seeds: {div_seeds}  |  successful seeds: {ok_seeds}")
    if not accs:
        print(f"{name:<24} -> ALL SEEDS DIVERGED ({dt:.0f}s)")
        return dict(name=name, acc_mean=None, diverged=n_div, total=len(SEEDS))
    am, as_ = float(np.mean(accs)) * 100, float(np.std(accs)) * 100
    lm, ls = float(np.mean(losses)), float(np.std(losses))
    n_valid = len(accs)
    caveat = "  [WARNING: n=1, std is meaningless]" if n_valid == 1 else ""
    print(f"{name:<24} -> acc={am:.2f}±{as_:.2f}%  loss={lm:.4f}±{ls:.4f}  "
          f"diverged={n_div}/{len(SEEDS)}  (n_valid={n_valid}){caveat}  ({dt:.0f}s)")
    return dict(name=name, acc_mean=am, acc_std=as_, loss_mean=lm, loss_std=ls,
                diverged=n_div, total=len(SEEDS))


def run_all():
    print("Loading data (CIFAR-10)...")
    train_set, test_set = load_data()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    results = []

    # --- Cycle-LIS only: repeat a stable small M, C times ---
    cycle_coeffs = cycle_lis_coeffs(CYCLE_M, CYCLE_C)
    cycle_steps = len(cycle_coeffs)
    print(f"Cycle-LIS: M={CYCLE_M}, C={CYCLE_C}  ->  total steps = {cycle_steps}, "
          f"batch_size={BATCH_SIZE}, number of seeds={len(SEEDS)}\n")
    results.append(run_method(f'Cycle-LIS (M={CYCLE_M},C={CYCLE_C})', train_set,
                               test_loader, cycle_steps, mode='cycle', coeffs=cycle_coeffs))

    print(f"\n{'='*70}")
    print(f"  {'Method':<24} {'Test Acc (%)':>16} {'Test Loss':>16} {'Diverged':>10}")
    print(f"  {'-'*66}")
    for r in results:
        if r.get('acc_mean') is None:
            print(f"  {r['name']:<24} {'--- diverged ---':>16}")
        else:
            print(f"  {r['name']:<24} {r['acc_mean']:.2f}±{r['acc_std']:.2f}"
                  f"{'':>4}{r['loss_mean']:.4f}±{r['loss_std']:.4f}{'':>4}"
                  f"{r['diverged']}/{r['total']}")
    print(f"{'='*70}")
    return results


if __name__ == "__main__":
    run_all()
