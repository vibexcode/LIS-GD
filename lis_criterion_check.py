# -*- coding: utf-8 -*-
"""
Kriter kontrolu: Delta L <= eps  ve  d_beta <= eps  (ikisi birden)
=====================================================================
Her veri seti icin, kuadratik modelde KAPALI FORMDA (simulasyon yok, CPU, saniyeler):
  * LIS-GD M_min : tam sayi cozunurlukte (geometrik izgara + bisection)
  * BGD iterasyon sayisi : e_n = (1 - eta*lam)^n * c0
her iki kriter icin:
  dL    = sum 0.5 * c0^2 * P^2 * lam        (Delta L = L(theta) - L(theta*))
  dbeta = sqrt(sum c0^2 * P^2)              (||theta - theta*||_2)
Rank eksik Hessian'da lam~0 yonleri icin c0 = 0 alinir (theta* = min-norm lstsq).

KULLANIM (Colab): bu dosyayi yukleyip `python lis_criterion_check.py`
Veri setleri: sklearn Diabetes + California Housing (indirmesiz/otomatik),
Bike Sharing (./data/day.csv, n=731), House Prices (./data/house_prices_train.csv),
Superconductivity (openml). Olmayan set atlanir. Advertising / Medical Insurance
icin EK_CSV sozlugune yol ekle (son sutun hedef degilse TARGET adini ver).
BGD adim buyuklugu: ETA_MODE ile sec (uc secenek hepsi yazdirilir).
"""
import os
import numpy as np

EPS = 1e-8
MAX_M = 2_000_000


def i_extra(k, M):
    k = np.asarray(k, dtype=np.float64)
    return np.log(k) / np.log(M) - k / M


def log_abs_P(lam, M):
    """log |P_i(M-2;M)| her ozdeger icin (P_i(M-1)=P_i(M-2) cunku k=M carpani 1)."""
    ks = np.arange(2, M)
    c = i_extra(ks, M)
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    out = np.zeros_like(lam)
    CH = 200000
    for s in range(0, len(ks), CH):
        cc = c[s:s + CH]
        out += np.log(np.abs(1.0 - np.outer(lam, cc)) + 1e-300).sum(axis=1)
    return out


def crit_vals(lam, c0, M):
    lp = np.clip(log_abs_P(lam, M), -700, 700)
    P2 = np.exp(2 * lp)
    dL = float(np.sum(0.5 * c0 ** 2 * P2 * lam))
    db = float(np.sqrt(np.sum(c0 ** 2 * P2)))
    return dL, db


def find_mmin(lam, c0, which):
    """Tam sayi M_min: geometrik izgarada ilk uygun nokta, sonra (onceki, bu] icinde bisection."""
    grid = np.unique(np.round(np.geomspace(6, MAX_M, 500)).astype(np.int64))
    idx = 0 if which == "dL" else 1
    prev = None
    for M in grid:
        v = crit_vals(lam, c0, int(M))[idx]
        if np.isfinite(v) and v <= EPS:
            lo = int(prev) if prev is not None else 5
            hi = int(M)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                vm = crit_vals(lam, c0, mid)[idx]
                if np.isfinite(vm) and vm <= EPS:
                    hi = mid
                else:
                    lo = mid
            return hi, int(M)  # (tam sayi M_min [bracket varsayimiyla], izgara degeri)
        prev = M
    return None, None


def bgd_iters(lam, c0, eta, which, nmax=10 ** 9):
    idx_f = (lambda n: np.sum(0.5 * c0 ** 2 * np.abs(1 - eta * lam) ** (2 * n) * lam)) if which == "dL" \
        else (lambda n: np.sqrt(np.sum(c0 ** 2 * np.abs(1 - eta * lam) ** (2 * n))))
    if np.max(np.abs(1 - eta * lam)) >= 1 and np.any((c0 != 0) & (np.abs(1 - eta * lam) >= 1)):
        return None
    hi = 1
    while hi < nmax and idx_f(hi) > EPS:
        hi *= 2
    if hi >= nmax:
        return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if idx_f(mid) > EPS:
            lo = mid
        else:
            hi = mid
    return hi


def analyse(X, y, name, whiten=False):
    n, d = X.shape
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    y = y - y.mean()
    tag = "whitened" if whiten else "raw"
    if whiten:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        keep = S > S.max() * max(n, d) * np.finfo(S.dtype).eps
        Z = U[:, keep] * np.sqrt(n)
        Xb = np.hstack([np.ones((n, 1)), Z])
    else:
        Xb = np.hstack([np.ones((n, 1)), X])
    H = (2.0 / n) * (Xb.T @ Xb)
    lam, V = np.linalg.eigh(H)
    lam = np.clip(lam, 0, None)
    th = np.linalg.lstsq(Xb, y, rcond=None)[0]
    c0 = V.T @ (0 - th)
    rank_tol = lam.max() * max(Xb.shape) * np.finfo(float).eps
    null = lam <= rank_tol
    c0 = np.where(null, 0.0, c0)
    print(f"\n=== {name} [{tag}] n={n} d_eff={Xb.shape[1]} rank={int((~null).sum())} "
          f"lam_min(>0)={lam[~null].min():.3e} lam_max={lam.max():.3e} kappa={lam.max()/lam[~null].min():.3e}")
    nz = ~null
    for which, lbl in (("dL", "Delta L<=1e-8"), ("db", "d_beta<=1e-8")):
        m, mg = find_mmin(lam[nz], c0[nz], which)
        print(f"  LIS-GD M_min [{lbl}] : {m}  (izgara ust siniri {mg})")
    lmax, lmin = lam.max(), lam[nz].min()
    for en, eta in (("1/lam_max", 1 / lmax), ("2/(lam_max+lam_min)", 2 / (lmax + lmin)), ("1/lam_max*0.5", 0.5 / lmax)):
        r = [bgd_iters(lam[nz], c0[nz], eta, w) for w in ("dL", "db")]
        print(f"  BGD eta={en:<22}: iter[Delta L]={r[0]}  iter[d_beta]={r[1]}")


def find(name):
    for p in (f"./{name}", f"./data/{name}", f"/content/{name}", f"/content/data/{name}"):
        if os.path.exists(p):
            return p
    return None


def main():
    from sklearn.datasets import load_diabetes, fetch_california_housing
    sets = []
    d = load_diabetes(); sets.append(("Diabetes", d.data.astype(float), d.target.astype(float)))
    try:
        c = fetch_california_housing(); sets.append(("California Housing", c.data.astype(float), c.target.astype(float)))
    except Exception as e:
        print("California atlandi:", e)
    try:
        import pandas as pd
        pday = find("day.csv")
        print("day.csv:", pday)
        if pday:
            df = pd.read_csv(pday)
            y = df["cnt"].values.astype(float)
            Xd = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"]).select_dtypes("number")
            sets.append(("Bike Sharing (day.csv)", Xd.values.astype(float), y))
        phouse = find("house_prices_train.csv")
        print("house_prices_train.csv:", phouse)
        if phouse:
            df = pd.read_csv(phouse)
            y = df["SalePrice"].values.astype(float)
            Xd = df.select_dtypes("number").drop(columns=["SalePrice", "Id"])
            Xd = Xd.fillna(Xd.median())
            sets.append(("House Prices", Xd.values.astype(float), y))
            # ek tani: hangi ozellikler dogrusal bagimli?
            Xs = (Xd.values - Xd.values.mean(0)) / (Xd.values.std(0) + 1e-12)
            _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
            tol = S.max() * max(Xs.shape) * np.finfo(float).eps
            for j in np.where(S <= tol)[0]:
                v = Vt[j]
                big = np.argsort(-np.abs(v))[:6]
                print("House Prices bagimlilik yonu:", [(Xd.columns[b], round(float(v[b]), 3)) for b in big if abs(v[b]) > 1e-3])
        # (ad, dosya yolu, hedef sutun) -- dosya yoksa atlanir
        EK = [("Medical Insurance", "./data/insurance.csv", "charges"),
              ("Advertising", "./data/advertising.csv", "Sales")]
        for nm, pth, tgt in EK:
            pth = find(os.path.basename(pth))
            if pth is None:
                print(f"{nm}: dosya bulunamadi, atlandi")
                continue
            print(f"{nm}: {pth}")
            df = pd.read_csv(pth)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            df = pd.get_dummies(df, drop_first=True).astype(float)
            sets.append((nm, df.drop(columns=[tgt]).values, df[tgt].values))
        # Energy Efficiency (UCI xlsx): X1..X8, hedef Y1 (heating) -- Y2 icin EE_TARGET degistir
        EE_TARGET = "Y1"
        for nm_ee in ("ENB2012_data.xlsx", "ENB2012_data.csv"):
            pth = find(nm_ee)
            if pth:
                print("Energy Efficiency:", pth)
                df = pd.read_excel(pth) if pth.endswith("xlsx") else pd.read_csv(pth)
                xs = [c for c in df.columns if str(c).startswith("X")]
                sets.append((f"Energy Efficiency ({EE_TARGET})", df[xs].values.astype(float),
                             df[EE_TARGET].values.astype(float)))
                break
        else:
            print("Energy Efficiency: dosya bulunamadi, atlandi")
        psc = find("superconductivity_train.csv")
        if psc:
            print("Superconductivity:", psc)
            sc = pd.read_csv(psc)
            sets.append(("Superconductivity", sc.drop(columns=["critical_temp"]).values.astype(float),
                         sc["critical_temp"].values.astype(float)))
        else:
            print("Superconductivity: superconductivity_train.csv bulunamadi, atlandi")
    except Exception as e:
        print("CSV yukleme hatasi:", e)
    for nm, X, y in sets:
        for w in (False, True):
            try:
                analyse(X.copy(), y.copy(), nm, whiten=w)
            except Exception as e:
                print(f"  {nm} [{'whitened' if w else 'raw'}] hata: {e}")


if __name__ == "__main__":
    main()
