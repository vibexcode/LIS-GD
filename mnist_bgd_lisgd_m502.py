"""
Colab'da calistir: MNIST uzerinde 784 -> 64 -> 10 MLP, tam-batch (mini-batch
degil) gradyan inisi. BGD (eta=0.1, 500 iterasyon) ve LIS-GD (M=502, yani
M-2=500 etkin guncelleme adimi -- BGD ile TAM ayni iterasyon sayisi) ayni
agirlik baslatmasiyla, ayni scriptte calistirilir, boylece iki sonuc
birebir ayni deneysel altyapidan gelir.

Gereksinim: sadece numpy ve (MNIST'i indirmek icin) tensorflow/keras ya da
sklearn. Asagidaki kod once keras.datasets.mnist'i dener, yoksa
sklearn fetch_openml'e duser.

Calistir: python mnist_bgd_lisgd_m502.py  (Colab hucresine yapistir)
"""

import numpy as np

SEED = 42


def load_mnist():
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(len(X_train), -1).astype(np.float64) / 255.0
        X_test = X_test.reshape(len(X_test), -1).astype(np.float64) / 255.0
        return X_train, y_train.astype(int), X_test, y_test.astype(int)
    except Exception:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        d = fetch_openml("mnist_784", version=1, as_frame=False)
        X = d.data.astype(np.float64) / 255.0
        y = d.target.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=10000, random_state=SEED, stratify=y)
        return X_train, y_train, X_test, y_test


def one_hot(y, K=10):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1
    return Y


def he_init(seed, d_in=784, d_hid=64, d_out=10):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(d_in, d_hid) * np.sqrt(2.0 / d_in)
    b1 = np.zeros(d_hid)
    W2 = rng.randn(d_hid, d_out) * np.sqrt(2.0 / d_hid)
    b2 = np.zeros(d_out)
    return [W1, b1, W2, b2]


def forward(params, X):
    W1, b1, W2, b2 = params
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU
    Z2 = A1 @ W2 + b2
    Z2 = Z2 - Z2.max(axis=1, keepdims=True)
    E = np.exp(Z2)
    P = E / E.sum(axis=1, keepdims=True)
    cache = (X, Z1, A1, P)
    return P, cache


def loss_and_grads(params, X, Y):
    W1, b1, W2, b2 = params
    n = X.shape[0]
    P, (X_, Z1, A1, _) = forward(params, X)
    eps = 1e-12
    loss = -np.mean(np.sum(Y * np.log(P + eps), axis=1))

    dZ2 = (P - Y) / n
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return loss, [dW1, db1, dW2, db2]


def accuracy(params, X, y):
    P, _ = forward(params, X)
    return np.mean(np.argmax(P, axis=1) == y)


def update(params, grads, step):
    return [p - step * g for p, g in zip(params, grads)]


def LIS(M):
    k = np.arange(2, M + 1)
    return np.log(k) / np.log(M) - k / M  # length M-1; last coeff (k=M) is 0 and dropped in run loop


def run_bgd(X_train, Y_train, X_test, y_test, eta, iters, seed):
    params = he_init(seed)
    for it in range(iters):
        loss, grads = loss_and_grads(params, X_train, Y_train)
        params = update(params, grads, eta)
    final_loss, _ = loss_and_grads(params, X_train, Y_train)
    test_acc = accuracy(params, X_test, y_test)
    return final_loss, test_acc, params


def run_lisgd(X_train, Y_train, X_test, y_test, M, seed):
    params = he_init(seed)
    coeffs = LIS(M)  # M-1 values; effective updates = M-2 per paper convention
    # per Algoritma 1 in the paper: k=2..M-1 used (k=M gives 0 and is skipped)
    coeffs = coeffs[:-1]  # drop the k=M term (=0), leaving M-2 effective steps
    for s in coeffs:
        loss, grads = loss_and_grads(params, X_train, Y_train)
        params = update(params, grads, s)
    final_loss, _ = loss_and_grads(params, X_train, Y_train)
    test_acc = accuracy(params, X_test, y_test)
    return final_loss, test_acc, params, len(coeffs)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    Y_train = one_hot(y_train)

    bgd_loss, bgd_acc, _ = run_bgd(X_train, Y_train, X_test, y_test, eta=0.1, iters=500, seed=SEED)
    print(f"BGD (eta=0.1, 500 iter): test_acc={bgd_acc:.4f}  train_loss={bgd_loss:.4f}")

    lis_loss, lis_acc, _, n_eff = run_lisgd(X_train, Y_train, X_test, y_test, M=502, seed=SEED)
    print(f"LIS-GD (M=502, {n_eff} etkin adim): test_acc={lis_acc:.4f}  train_loss={lis_loss:.4f}")