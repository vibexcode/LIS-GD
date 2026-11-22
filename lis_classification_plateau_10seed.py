"""
10-seed robustness for the Bolum 4.3 classification redo (Wine, Digits,
Breast Cancer Wisconsin), using training-accuracy-plateau M selection
(never touches test data), mirroring the 4.1/4.2 seed-robustness checks.

Run: python lis_classification_plateau_10seed.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def LIS(M):
    k = np.arange(2, M + 1)
    return np.log(k) / np.log(M) - k / M


def one_hot(y, K):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1
    return Y


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def ce_loss(theta, Xb, Y):
    P = softmax(Xb @ theta)
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))


def ce_grad(theta, Xb, Y):
    P = softmax(Xb @ theta)
    n = Xb.shape[0]
    return Xb.T @ (P - Y) / n


def accuracy(theta, Xb, y):
    P = softmax(Xb @ theta)
    return np.mean(np.argmax(P, axis=1) == y)


def run_bgd(Xb, Y, K, eta, max_iter=5000, tol=1e-9):
    theta = np.zeros((Xb.shape[1], K))
    prev = None
    for it in range(1, max_iter + 1):
        loss = ce_loss(theta, Xb, Y)
        theta = theta - eta * ce_grad(theta, Xb, Y)
        if prev is not None and abs(prev - loss) < tol:
            break
        prev = loss
    return theta, ce_loss(theta, Xb, Y), it


def lis_run(Xb, Y, K, M):
    theta = np.zeros((Xb.shape[1], K))
    for s in LIS(M):
        theta = theta - s * ce_grad(theta, Xb, Y)
        if not np.all(np.isfinite(theta)):
            return theta, np.inf
    return theta, ce_loss(theta, Xb, Y)


def find_M_by_train_acc_plateau(Xb, Y, y_train, M_max, growth=1.3, patience=3):
    K = Y.shape[1]
    M = 10
    best_acc = -1
    stale = 0
    chosen_M = M
    while M < M_max:
        theta, _ = lis_run(Xb, Y, K, M)
        acc = accuracy(theta, Xb, y_train)
        if acc > best_acc + 1e-9:
            best_acc = acc
            chosen_M = M
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                return chosen_M, best_acc
        M = min(int(M * growth) + 1, M_max)
    return chosen_M, best_acc


def run_seed(X, y, eta, seed, M_max, bgd_max_iter):
    K = len(np.unique(y))
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_raw)
    X_test = sc.transform(X_test_raw)
    Xb_train = np.c_[np.ones(len(X_train)), X_train]
    Xb_test = np.c_[np.ones(len(X_test)), X_test]
    Y_train = one_hot(y_train, K)

    theta_bgd, bgd_train_loss, bgd_iters = run_bgd(Xb_train, Y_train, K, eta, max_iter=bgd_max_iter)
    bgd_test_acc = accuracy(theta_bgd, Xb_test, y_test)

    M_star, _ = find_M_by_train_acc_plateau(Xb_train, Y_train, y_train, M_max=M_max)
    theta_lis, _ = lis_run(Xb_train, Y_train, K, M_star)
    lis_test_acc = accuracy(theta_lis, Xb_test, y_test)

    savings = 100 * (1 - (M_star - 1) / bgd_iters)
    return dict(seed=seed, bgd_iters=bgd_iters, bgd_test_acc=bgd_test_acc,
                M_star=M_star, lis_test_acc=lis_test_acc, savings=savings)


def analyze(name, X, y, eta, seeds, M_max, bgd_max_iter):
    rows = [run_seed(X, y, eta, s, M_max, bgd_max_iter) for s in seeds]
    df = pd.DataFrame(rows)
    print(f"\n{name}")
    print(df.to_string(index=False))
    print("Summary (mean +/- std):")
    for col in ["bgd_iters", "bgd_test_acc", "M_star", "lis_test_acc", "savings"]:
        print(f"  {col}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
    return df


if __name__ == "__main__":
    seeds = list(range(10))

    wine = load_wine()
    analyze("Wine", wine.data, wine.target, eta=0.5, seeds=seeds, M_max=2000, bgd_max_iter=5000)

    digits = load_digits()
    analyze("Digits", digits.data, digits.target, eta=0.3, seeds=seeds, M_max=2000, bgd_max_iter=6000)

    bc = load_breast_cancer()
    analyze("Breast Cancer Wisconsin", bc.data, bc.target, eta=0.5, seeds=seeds, M_max=2000, bgd_max_iter=5000)
