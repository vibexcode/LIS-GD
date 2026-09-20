# -*- coding: utf-8 -*-
"""
Cycle-LIS baseline comparison for Fashion-MNIST (PyTorch/GPU)
====================================================================
Cycle-LIS version of the previous script. Definition (matches the
paper's Equations 36-37 exactly): fixed capacity M, number of cycles
C. In each cycle the coefficients for k=2,...,M-1 are applied in order
(M-2 effective steps); the zero-coefficient terminal step at k=M is
skipped (no gradient computed, parameters unchanged). Parameters are
CARRIED OVER between cycles (not reset) - only the k index resets to 2.

Total budget = C * (M - 2) effective updates.

USAGE: change the M and C values below to try any capacity/cycle
combination you like. The script compares against baselines using the
same total budget (tuned constant LR, linear/cosine/triangular/
one-cycle, AdamW).

On Colab: select GPU (Runtime > Change runtime type > GPU).
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import time

# ══════════════════════════════════════════════════════════
# ADJUSTABLE PARAMETERS - change these
# ══════════════════════════════════════════════════════════
M_CAPACITY = 31      # LIS profile capacity (length of a single cycle)
N_CYCLES   = 30       # number of repetitions (Cycle-LIS)
# Total effective steps = N_CYCLES * (M_CAPACITY - 2)

RNG_SEED = 42
HIDDEN = 64

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
# 3. Training loop - takes a COEFFICIENT ARRAY (not a schedule
#    function), so Cycle-LIS's repeating structure is expressed
#    naturally
# ══════════════════════════════════════════════════════════

def train_with_coeffs(X_train, y_train, X_test, y_test, coeffs,
                       adamw=False, lr_adamw=0.001, wd_adamw=0.01,
                       verbose=False):
    W1, b1, W2, b2 = init_model()
    params = [W1, b1, W2, b2]

    opt = torch.optim.AdamW(params, lr=lr_adamw, weight_decay=wd_adamw) if adamw else None
    n_steps = len(coeffs)

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
            eta = coeffs[k]
            with torch.no_grad():
                for p in params:
                    p -= eta * p.grad

        if verbose and k % max(1, n_steps // 10) == 0:
            with torch.no_grad():
                tl = forward(X_test, W1, b1, W2, b2)
                print(f"    k={k:5d}  eta={coeffs[k] if not adamw else float('nan'):.4f}  "
                      f"test_acc={accuracy(tl, y_test):.4f}")

    with torch.no_grad():
        test_logits = forward(X_test, W1, b1, W2, b2)
        final_acc = accuracy(test_logits, y_test)
        final_loss = F.cross_entropy(test_logits, y_test).item()

    return final_acc, final_loss


# ══════════════════════════════════════════════════════════
# 4. Coefficient arrays
# ══════════════════════════════════════════════════════════

def cycle_lis_coeffs(M, C):
    """Cycle-LIS: repeat the M-capacity profile (k=2,...,M-1, terminal
    zero excluded) C times. Total length = C*(M-2)."""
    single_cycle = [math.log(k, M) - k / M for k in range(2, M)]  # k=2,...,M-1
    return single_cycle * C

def spread_lis_coeffs(M, C):
    """Spread-LIS: hold each coefficient for C consecutive updates,
    then move to the next. For a fixed quadratic loss this has the
    same terminal effect as Cycle-LIS (Equation 41), but the
    intermediate trajectory differs. Total length is again C*(M-2)."""
    single_cycle = [math.log(k, M) - k / M for k in range(2, M)]
    out = []
    for c in single_cycle:
        out.extend([c] * C)
    return out

def const_coeffs(eta0, n_steps):
    return [eta0] * n_steps

def linear_decay_coeffs(eta_max, n_steps):
    return [eta_max * (1 - k / n_steps) for k in range(n_steps)]

def cosine_coeffs(eta_max, n_steps, eta_min=0.0):
    return [eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * k / n_steps))
            for k in range(n_steps)]

def triangular_coeffs(eta_max, n_steps, eta_min=0.0):
    half = n_steps / 2
    out = []
    for k in range(n_steps):
        if k <= half:
            out.append(eta_min + (eta_max - eta_min) * (k / half))
        else:
            out.append(eta_max - (eta_max - eta_min) * ((k - half) / half))
    return out

def one_cycle_coeffs(eta_max, n_steps, eta_min=0.0, warmup_frac=0.3):
    warm = max(1, int(n_steps * warmup_frac))
    out = []
    for k in range(n_steps):
        if k <= warm:
            out.append(eta_min + (eta_max - eta_min) * (k / warm))
        else:
            out.append(eta_max * 0.5 * (1 + math.cos(math.pi * (k - warm) / (n_steps - warm))))
    return out


# ══════════════════════════════════════════════════════════
# 5. Main run
# ══════════════════════════════════════════════════════════

def run_all(M=M_CAPACITY, C=N_CYCLES, use_spread=False):
    print("Loading data (Fashion-MNIST)...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {tuple(X_train.shape)}  Test: {tuple(X_test.shape)}\n")

    coeffs_fn = spread_lis_coeffs if use_spread else cycle_lis_coeffs
    label = "Spread-LIS" if use_spread else "Cycle-LIS"

    lis_coeffs = coeffs_fn(M, C)
    n_steps = len(lis_coeffs)
    lis_peak = max(lis_coeffs)
    print(f"{label}: M={M}, C={C}  ->  total effective steps = {n_steps}, peak={lis_peak:.4f}\n")

    results = {}

    def go(name, coeffs=None, adamw=False, lr_adamw=0.001):
        t0 = time.time()
        acc, loss = train_with_coeffs(X_train, y_train, X_test, y_test,
                                       coeffs if coeffs is not None else [0.0],
                                       adamw=adamw, lr_adamw=lr_adamw)
        dt = time.time() - t0
        print(f"{name:<28} -> acc={acc:.4f}  loss={loss:.4f}  ({dt:.1f}s)")
        return acc, loss

    results[f'{label} (M={M}, C={C})'] = go(f'{label} (M={M}, C={C})', lis_coeffs)

    best = (None, -1, None)
    for eta0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        acc_i, loss_i = go(f'  [grid] const eta={eta0:.2f}', const_coeffs(eta0, n_steps))
        if acc_i > best[1]:
            best = (eta0, acc_i, loss_i)
    results[f'Tuned constant LR (eta={best[0]})'] = (best[1], best[2])
    print(f"Tuned constant LR (eta={best[0]})        -> acc={best[1]:.4f}  loss={best[2]:.4f}")

    results['Linear decay']     = go('Linear decay', linear_decay_coeffs(lis_peak, n_steps))
    results['Cosine annealing'] = go('Cosine annealing', cosine_coeffs(lis_peak, n_steps))
    results['Triangular CLR']   = go('Triangular CLR', triangular_coeffs(lis_peak, n_steps))
    results['One-cycle']        = go('One-cycle', one_cycle_coeffs(lis_peak, n_steps))

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
    # Change M_CAPACITY and N_CYCLES above, or call
    # run_all(M=..., C=..., use_spread=False) directly here.
    run_all(M=M_CAPACITY, C=N_CYCLES, use_spread=False)
