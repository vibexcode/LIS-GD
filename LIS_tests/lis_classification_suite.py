# -*- coding: utf-8 -*-
"""
Classification suite - canonical LIS-GD verification
=======================================================
Compares canonical LIS-GD to sklearn's own classifier and to tuned
constant-LR gradient descent on Breast Cancer Wisconsin (binary,
logistic regression), Wine (3 classes, softmax regression), and
Digits (10 classes, softmax regression). All small datasets, seconds
on CPU.

Independently verifies the Breast Cancer claim in the source notes
(sklearn 97.37% vs LIS(1000) 98.25%). NOTE: sklearn's default
LogisticRegression applies L2 regularization; for a fair comparison
against unregularized LIS-GD, also compare against an unregularized
sklearn reference (see the notes recorded during the analysis).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 42
EPS = 1e-8


def i_extra(k, M):
    return np.log(k) / np.log(M) - k / M


# ══════════════════════════════════════════════════════════
# Binary logistic regression (Breast Cancer)
# ══════════════════════════════════════════════════════════

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def bce_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def run_logreg_lis(X, y, M):
    n, d = X.shape
    theta = np.zeros(d)
    for k in range(2, M):
        p = sigmoid(X @ theta)
        grad = (1.0 / n) * X.T @ (p - y)
        theta = theta - i_extra(k, M) * grad
    return theta

def run_logreg_const(X, y, eta, n_iters):
    n, d = X.shape
    theta = np.zeros(d)
    for _ in range(n_iters):
        p = sigmoid(X @ theta)
        grad = (1.0 / n) * X.T @ (p - y)
        theta = theta - eta * grad
        if not np.all(np.isfinite(theta)):
            return theta, True
    return theta, False


def test_breast_cancer():
    print(f"\n{'='*60}\nBreast Cancer Wisconsin (binary classification)\n{'='*60}")
    data = load_breast_cancer()
    X_raw, y = data.data.astype(np.float64), data.target.astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler().fit(X_train)
    Xtr = np.hstack([np.ones((len(X_train), 1)), scaler.transform(X_train)])
    Xte = np.hstack([np.ones((len(X_test), 1)), scaler.transform(X_test)])

    # sklearn reference (default: L2-regularized, C=1.0)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    acc_sklearn = clf.score(scaler.transform(X_test), y_test)
    print(f"sklearn LogisticRegression (default, C=1.0): test_acc={acc_sklearn*100:.2f}%")

    # sklearn reference, UNREGULARIZED (fair comparison to LIS-GD/const-LR,
    # neither of which uses any regularization)
    clf_unreg = LogisticRegression(max_iter=5000, C=1e6).fit(scaler.transform(X_train), y_train)
    acc_unreg = clf_unreg.score(scaler.transform(X_test), y_test)
    print(f"sklearn LogisticRegression (unregularized, C->inf): test_acc={acc_unreg*100:.2f}%")

    # canonical LIS-GD, a few M values
    for M in [200, 500, 1000]:
        theta = run_logreg_lis(Xtr, y_train, M)
        p_test = sigmoid(Xte @ theta)
        acc = np.mean((p_test >= 0.5) == y_test)
        loss = bce_loss(p_test, y_test)
        print(f"LIS-GD (M={M}, {M-2} effective steps): test_acc={acc*100:.2f}%  "
              f"test_BCE={loss:.4f}")

    # tuned constant LR (same budget as M=1000's effective step count)
    n_iters = 998
    best_eta, best_acc = None, -1
    for eta in [0.01, 0.05, 0.1, 0.3, 0.5]:
        theta, div = run_logreg_const(Xtr, y_train, eta, n_iters)
        if div:
            continue
        p_test = sigmoid(Xte @ theta)
        acc = np.mean((p_test >= 0.5) == y_test)
        if acc > best_acc:
            best_eta, best_acc = eta, acc
    print(f"Tuned constant LR (eta={best_eta}, {n_iters} steps): test_acc={best_acc*100:.2f}%")


# ══════════════════════════════════════════════════════════
# Multi-class softmax regression (Wine, Digits)
# ══════════════════════════════════════════════════════════

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1
    return Y

def ce_loss(P, Y):
    return -np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1)), axis=1))

def run_softmax_lis(X, Y, M):
    n, d = X.shape
    k_classes = Y.shape[1]
    Theta = np.zeros((d, k_classes))
    for k in range(2, M):
        P = softmax(X @ Theta)
        grad = (1.0 / n) * X.T @ (P - Y)
        Theta = Theta - i_extra(k, M) * grad
    return Theta

def run_softmax_const(X, Y, eta, n_iters):
    n, d = X.shape
    k_classes = Y.shape[1]
    Theta = np.zeros((d, k_classes))
    for _ in range(n_iters):
        P = softmax(X @ Theta)
        grad = (1.0 / n) * X.T @ (P - Y)
        Theta = Theta - eta * grad
        if not np.all(np.isfinite(Theta)):
            return Theta, True
    return Theta, False


def test_multiclass(name, loader, M_list):
    print(f"\n{'='*60}\n{name} (multi-class classification)\n{'='*60}")
    data = loader()
    X_raw, y = data.data.astype(np.float64), data.target.astype(np.int64)
    n_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler().fit(X_train)
    Xtr = np.hstack([np.ones((len(X_train), 1)), scaler.transform(X_train)])
    Xte = np.hstack([np.ones((len(X_test), 1)), scaler.transform(X_test)])
    Ytr = one_hot(y_train, n_classes)
    Yte = one_hot(y_test, n_classes)

    clf = LogisticRegression(max_iter=1000).fit(
        scaler.transform(X_train), y_train)
    acc_sklearn = clf.score(scaler.transform(X_test), y_test)
    print(f"sklearn LogisticRegression: test_acc={acc_sklearn*100:.2f}%")

    for M in M_list:
        Theta = run_softmax_lis(Xtr, Ytr, M)
        P_test = softmax(Xte @ Theta)
        acc = np.mean(P_test.argmax(1) == y_test)
        loss = ce_loss(P_test, Yte)
        print(f"LIS-GD (M={M}, {M-2} effective steps): test_acc={acc*100:.2f}%  "
              f"test_CE={loss:.4f}")

    n_iters = M_list[-1] - 2
    best_eta, best_acc = None, -1
    for eta in [0.01, 0.05, 0.1, 0.3, 0.5]:
        Theta, div = run_softmax_const(Xtr, Ytr, eta, n_iters)
        if div:
            continue
        P_test = softmax(Xte @ Theta)
        acc = np.mean(P_test.argmax(1) == y_test)
        if acc > best_acc:
            best_eta, best_acc = eta, acc
    print(f"Tuned constant LR (eta={best_eta}, {n_iters} steps): test_acc={best_acc*100:.2f}%")


if __name__ == "__main__":
    test_breast_cancer()
    test_multiclass("Wine", load_wine, [50, 100, 200])
    test_multiclass("Digits", load_digits, [100, 300, 500])
