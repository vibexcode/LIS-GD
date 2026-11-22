# -*- coding: utf-8 -*-
"""
Tablo 4 / Tablo S9 icin TEK KRITER: d_beta = ||theta - theta_OLS||_2 <= 1e-8
Superconductivity, Bike Sharing, House Prices (ham ve beyazlatilmis).
Kapali formda, CPU, saniyeler-dakikalar. Dosyalar /content/ altinda:
  superconductivity_train.csv, day.csv, house_prices_train.csv
Cikti: BGD (eta=0.1/0.3/0.5/1/lam_max) iterasyonlari + LIS-GD M_min (tam sayi).
"""
import os
import numpy as np

EPS = 1e-8
MAX_M = 2_000_000
BUDGET = 10_000_000          # BGD iterasyon butcesi


def find(name):
    for p in (f"./{name}", f"./data/{name}", f"/content/{name}", f"/content/data/{name}"):
        if os.path.exists(p):
            return p
    return None


def i_extra(k, M):
    k = np.asarray(k, dtype=np.float64)
    return np.log(k) / np.log(M) - k / M


def dbeta_lis(lam, c0, M):
    ks = np.arange(2, M)
    c = i_extra(ks, M)
    out = np.zeros_like(lam)
    for s in range(0, len(ks), 200000):
        out += np.log(np.abs(1.0 - np.outer(lam, c[s:s + 200000])) + 1e-300).sum(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.sqrt(np.sum(c0 ** 2 * np.exp(2 * np.clip(out, -700, 700)))))


def lis_mmin(lam, c0):
    grid = np.unique(np.round(np.geomspace(6, MAX_M, 500)).astype(np.int64))
    prev = None
    for M in grid:
        v = dbeta_lis(lam, c0, int(M))
        if np.isfinite(v) and v <= EPS:
            lo, hi = (int(prev) if prev is not None else 5), int(M)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                vm = dbeta_lis(lam, c0, mid)
                if np.isfinite(vm) and vm <= EPS:
                    hi = mid
                else:
                    lo = mid
            return hi
        prev = M
    return None


def bgd_iters(lam, c0, eta):
    r = np.abs(1 - eta * lam)
    if np.any(r >= 1 - 1e-15):
        return "Iraksar"
    f = lambda n: float(np.sqrt(np.sum(c0 ** 2 * r ** (2.0 * n))))
    if f(BUDGET) > EPS:
        return "Butce asildi (>1e7)"
    hi = 1
    while f(hi) > EPS:
        hi *= 2
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) > EPS:
            lo = mid
        else:
            hi = mid
    return hi


def analyse(X, y, name, whiten):
    n, d = X.shape
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    y = y - y.mean()
    if whiten:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        keep = S > S.max() * max(n, d) * np.finfo(S.dtype).eps
        Xb = np.hstack([np.ones((n, 1)), U[:, keep] * np.sqrt(n)])
    else:
        Xb = np.hstack([np.ones((n, 1)), X])
    H = (2.0 / n) * (Xb.T @ Xb)
    lam, V = np.linalg.eigh(H)
    th = np.linalg.lstsq(Xb, y, rcond=None)[0]
    c0 = V.T @ (0 - th)
    nz = lam > lam.max() * max(Xb.shape) * np.finfo(float).eps
    lam, c0 = lam[nz], c0[nz]
    kappa = lam.max() / lam.min()
    lmax = lam.max()
    print(f"\n{name} [{'Beyazlatilmis' if whiten else 'Ham'}]  n={n}  rank={nz.sum()}/{len(nz)}  "
          f"lam_max={lmax:.4g}  kappa(sifirdan farkli ozdegerler)={kappa:.5g}")
    res = {f"eta={e}": bgd_iters(lam, c0, e) for e in (0.1, 0.3, 0.5)}
    res["eta=1/lam_max"] = bgd_iters(lam, c0, 1 / lmax)
    print("  BGD:", res)
    m = lis_mmin(lam, c0)
    print("  LIS-GD M_min (d_beta<=1e-8):", m if m is not None else "Tanimsiz")


def main():
    import pandas as pd
    sets = []
    p = find("superconductivity_train.csv")
    if p:
        sc = pd.read_csv(p)
        sets.append(("Superconductivity", sc.drop(columns=["critical_temp"]).values.astype(float),
                     sc["critical_temp"].values.astype(float)))
    p = find("day.csv")
    if p:
        df = pd.read_csv(p)
        Xd = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"]).select_dtypes("number")
        sets.append(("Bike Sharing", Xd.values.astype(float), df["cnt"].values.astype(float)))
    p = find("house_prices_train.csv")
    if p:
        df = pd.read_csv(p)
        Xd = df.select_dtypes("number").drop(columns=["SalePrice", "Id"])
        Xd = Xd.fillna(Xd.median())
        sets.append(("House Prices", Xd.values.astype(float), df["SalePrice"].values.astype(float)))
    for nm, X, y in sets:
        for w in (False, True):
            analyse(X.copy(), y.copy(), nm, w)


if __name__ == "__main__":
    main()