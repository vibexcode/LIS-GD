"""
Colab'da calistir (GPU runtime): TEK script icinde hem BGD (8 eta) hem
LIS-GD, ayni 7 butce (500,1000,2000,3000,4000,5000,10000) ve ayni 5 tohum
({42,7,123,2024,31337}) uzerinden. Baslat, bilgisayar basinda beklemene
gerek yok -- bitince tek bir birlesik karsilastirma tablosu basar.

Her (eta, tohum) cifti icin TEK 10.000 iterasyonluk BGD egitimi calisir,
istenen butce noktalarinda checkpoint alinir. LIS-GD icin her butceye
karsilik gelen M ayri egitilir (M = butce + 2, boylece M-2 etkin adim
tam olarak butceye esit olur).

Calistir: python mnist_full_comparison_gpu.py  (Colab hucresine yapistir)
"""

import numpy as np
import torch

SEEDS = [42, 7, 123, 2024, 31337]
ETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
CHECKPOINTS = [500, 1000, 2000, 3000, 4000, 5000, 10000]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanilan cihaz:", device)


def load_mnist():
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
        X_test = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0
        return X_train, y_train.astype(np.int64), X_test, y_test.astype(np.int64)
    except Exception:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        d = fetch_openml("mnist_784", version=1, as_frame=False)
        X = (d.data.astype(np.float32)) / 255.0
        y = d.target.astype(np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=10000, random_state=42, stratify=y)
        return X_train, y_train, X_test, y_test


def he_init(seed, d_in=784, d_hid=64, d_out=10):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W1 = (torch.randn(d_in, d_hid, generator=g) * np.sqrt(2.0 / d_in)).to(device)
    b1 = torch.zeros(d_hid, device=device)
    W2 = (torch.randn(d_hid, d_out, generator=g) * np.sqrt(2.0 / d_hid)).to(device)
    b2 = torch.zeros(d_out, device=device)
    return [W1, b1, W2, b2]


def forward(params, X):
    W1, b1, W2, b2 = params
    Z1 = X @ W1 + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2 + b2
    return Z2, (X, Z1, A1)


def loss_and_grads(params, X, Y_onehot):
    W1, b1, W2, b2 = params
    n = X.shape[0]
    Z2, (X_, Z1, A1) = forward(params, X)
    logP = torch.log_softmax(Z2, dim=1)
    loss = -(Y_onehot * logP).sum(dim=1).mean()
    P = torch.softmax(Z2, dim=1)

    dZ2 = (P - Y_onehot) / n
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(dim=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0).float()
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(dim=0)
    return loss.item(), [dW1, db1, dW2, db2]


def eval_loss(params, X, Y_onehot):
    with torch.no_grad():
        Z2, _ = forward(params, X)
        logP = torch.log_softmax(Z2, dim=1)
        return (-(Y_onehot * logP).sum(dim=1).mean()).item()


def accuracy(params, X, y):
    Z2, _ = forward(params, X)
    pred = torch.argmax(Z2, dim=1)
    return (pred == y).float().mean().item()


def is_finite_params(params):
    return all(torch.isfinite(p).all().item() for p in params)


def LIS(M):
    k = np.arange(2, M + 1)
    return (np.log(k) / np.log(M) - k / M)[:-1]  # drop k=M term (=0); M-2 effective steps


def run_bgd_with_checkpoints(X_train, Y_onehot, X_test, Y_test_onehot, y_test, eta, seed, checkpoints):
    params = he_init(seed)
    results = {}  # it -> (train_loss, test_loss, test_acc)
    cp_set = set(checkpoints)
    max_iter = max(checkpoints)
    diverged = False
    for it in range(1, max_iter + 1):
        loss, grads = loss_and_grads(params, X_train, Y_onehot)
        with torch.no_grad():
            params = [p - eta * g for p, g in zip(params, grads)]
        if not is_finite_params(params):
            diverged = True
            break
        if it in cp_set:
            train_loss = eval_loss(params, X_train, Y_onehot)
            test_loss = eval_loss(params, X_test, Y_test_onehot)
            acc = accuracy(params, X_test, y_test)
            results[it] = (train_loss, test_loss, acc)
    if diverged:
        for cp in checkpoints:
            if cp not in results:
                results[cp] = (None, None, None)
    return results


def run_lisgd(X_train, Y_onehot, X_test, Y_test_onehot, y_test, M, seed):
    params = he_init(seed)
    for s in LIS(M):
        loss, grads = loss_and_grads(params, X_train, Y_onehot)
        with torch.no_grad():
            params = [p - float(s) * g for p, g in zip(params, grads)]
    train_loss = eval_loss(params, X_train, Y_onehot)
    test_loss = eval_loss(params, X_test, Y_test_onehot)
    return train_loss, test_loss, accuracy(params, X_test, y_test)


def fmt_row(label, vals):
    accs = [a for (tl, xl, a) in vals if a is not None]
    trlosses = [tl for (tl, xl, a) in vals if tl is not None]
    telosses = [xl for (tl, xl, a) in vals if xl is not None]
    div = sum(1 for (tl, xl, a) in vals if a is None)
    if accs:
        return (f"{label:<16}{np.mean(accs)*100:6.2f} +/- {np.std(accs)*100:5.2f}%   "
                f"{np.mean(trlosses):.4f} +/- {np.std(trlosses):.4f}      "
                f"{np.mean(telosses):.4f} +/- {np.std(telosses):.4f}      {div}/{len(SEEDS)}")
    return f"{label:<16}--- hepsi ıraksadı ---                                                    {div}/{len(SEEDS)}"


if __name__ == "__main__":
    X_train_np, y_train_np, X_test_np, y_test_np = load_mnist()
    X_train = torch.tensor(X_train_np, device=device)
    X_test = torch.tensor(X_test_np, device=device)
    y_test = torch.tensor(y_test_np, device=device)
    Y_onehot = torch.zeros(len(y_train_np), 10, device=device)
    Y_onehot[torch.arange(len(y_train_np)), torch.tensor(y_train_np)] = 1.0
    Y_test_onehot = torch.zeros(len(y_test_np), 10, device=device)
    Y_test_onehot[torch.arange(len(y_test_np)), torch.tensor(y_test_np)] = 1.0

    header = (f"{'yontem':<16}{'test_acc (ort+/-std)':<24}"
              f"{'train_loss (ort+/-std)':<26}{'test_loss (ort+/-std)':<26}{'Iraksama'}")

    # ---- BGD: her (eta,tohum) icin tek 10000-iterasyonluk kosu, checkpointli ----
    bgd_results = {}  # (eta,budget) -> list[(train_loss,test_loss,acc)]
    for eta in ETAS:
        for seed in SEEDS:
            res = run_bgd_with_checkpoints(X_train, Y_onehot, X_test, Y_test_onehot, y_test, eta, seed, CHECKPOINTS)
            for cp, triple in res.items():
                bgd_results.setdefault((eta, cp), []).append(triple)
        print(f"BGD eta={eta} tamamlandi")

    # ---- LIS-GD: her butceye esit M = butce+2 ----
    lis_results = {}  # budget -> list[(train_loss,test_loss,acc)]
    for cp in CHECKPOINTS:
        M = cp + 2
        for seed in SEEDS:
            triple = run_lisgd(X_train, Y_onehot, X_test, Y_test_onehot, y_test, M, seed)
            lis_results.setdefault(cp, []).append(triple)
        print(f"LIS-GD butce={cp} (M={M}) tamamlandi")

    # ---- Birlesik tablo ----
    print(f"\n{header}")
    for cp in CHECKPOINTS:
        print(f"\n-- butce = {cp} --")
        for eta in ETAS:
            print(fmt_row(f"BGD eta={eta}", bgd_results.get((eta, cp), [])))
        print(fmt_row("LIS-GD", lis_results.get(cp, [])))