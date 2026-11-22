"""
Colab'da calistir: MNIST uzerinde 784 -> 64 -> 10 MLP, tam-batch gradyan
inisi. Bolum 4.5'in geri kalaniyla ayni titizlikte: BGD icin birden fazla
ogrenme hizi (0.05 / 0.1 / 0.2 / 0.3) VE Fashion-MNIST'te kullanilan ayni
5 tohum ({42,7,123,2024,31337}) uzerinden ortalama +/- std; LIS-GD (M=502,
500 etkin adim, ogrenme hizindan bagimsiz) de ayni 5 tohumla.

Herkes TAM OLARAK ayni iterasyon butcesinde (500 guncelleme adimi)
calisir, boylece "hangi eta'yi sectiniz" sorusuna cevap verir.

Not: BGD yuksek eta'da (orn. 0.3+) ıraksayabilir; kod NaN/Inf durumunu
"Iraksadi" olarak isaretler.

Calistir: python mnist_seed_lr_grid.py  (Colab hucresine yapistir)
"""

import numpy as np

SEEDS = [42, 7, 123, 2024, 31337]
ETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
BUDGET = 500  # effective update steps for both methods


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
            X, y, test_size=10000, random_state=42, stratify=y)
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
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    Z2 = Z2 - Z2.max(axis=1, keepdims=True)
    E = np.exp(Z2)
    P = E / E.sum(axis=1, keepdims=True)
    return P, (X, Z1, A1)


def loss_and_grads(params, X, Y):
    W1, b1, W2, b2 = params
    n = X.shape[0]
    P, (X_, Z1, A1) = forward(params, X)
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


def is_finite_params(params):
    return all(np.all(np.isfinite(p)) for p in params)


def run_bgd(X_train, Y_train, X_test, y_test, eta, iters, seed):
    params = he_init(seed)
    for it in range(iters):
        loss, grads = loss_and_grads(params, X_train, Y_train)
        params = [p - eta * g for p, g in zip(params, grads)]
        if not is_finite_params(params):
            return None, None
    final_loss, _ = loss_and_grads(params, X_train, Y_train)
    return final_loss, accuracy(params, X_test, y_test)


def LIS(M):
    k = np.arange(2, M + 1)
    return (np.log(k) / np.log(M) - k / M)[:-1]  # drop k=M term (=0)


def run_lisgd(X_train, Y_train, X_test, y_test, M, seed):
    params = he_init(seed)
    for s in LIS(M):
        loss, grads = loss_and_grads(params, X_train, Y_train)
        params = [p - s * g for p, g in zip(params, grads)]
    final_loss, _ = loss_and_grads(params, X_train, Y_train)
    return final_loss, accuracy(params, X_test, y_test)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    Y_train = one_hot(y_train)

    print(f"{'Yontem':<20}{'test_acc (ort+/-std)':<28}{'train_loss (ort+/-std)':<28}{'Iraksama'}")

    for eta in ETAS:
        accs, losses, div = [], [], 0
        for seed in SEEDS:
            loss, acc = run_bgd(X_train, Y_train, X_test, y_test, eta, BUDGET, seed)
            if loss is None:
                div += 1
            else:
                accs.append(acc)
                losses.append(loss)
        if accs:
            print(f"BGD (eta={eta:<5}) {np.mean(accs)*100:6.2f} +/- {np.std(accs)*100:5.2f}%      "
                  f"{np.mean(losses):.4f} +/- {np.std(losses):.4f}          {div}/{len(SEEDS)}")
        else:
            print(f"BGD (eta={eta:<5})  --- hepsi ıraksadı ---                                      {div}/{len(SEEDS)}")

    accs, losses = [], []
    for seed in SEEDS:
        loss, acc = run_lisgd(X_train, Y_train, X_test, y_test, M=502, seed=seed)
        accs.append(acc)
        losses.append(loss)
    print(f"LIS-GD (M=502)     {np.mean(accs)*100:6.2f} +/- {np.std(accs)*100:5.2f}%      "
          f"{np.mean(losses):.4f} +/- {np.std(losses):.4f}          0/{len(SEEDS)}")