# LIS-GD: Finite-Horizon Gradient Optimization

Code and experiments accompanying the paper
**"LIS-GD: Finite-Horizon Gradient Optimization"** (Turkish title: "LIS-GD: Sonlu Ufuklu
Gradyan Optimizasyonu").

LIS-GD is a gradient-descent method whose per-step coefficients are derived
in closed form from a single horizon parameter **M** via the *Logarithmic
Information Surplus* (LIS) profile:

```
I_extra(k; M) = ln(k) / ln(M) - k / M,     k = 2, ..., M
theta_{n+1}   = theta_n - I_extra(n+2; M) * grad L(theta_n),   n = 0, ..., M-2
```

The coefficient rises, peaks once, and returns to exactly zero at k = M,
so the optimizer terminates by construction, with no separate stopping rule
and no external learning-rate schedule. It carries no extra state (no
momentum, no second-moment estimates).

The paper analyzes this schedule on quadratic models and evaluates it
empirically on a controlled synthetic regression problem and thirteen
real-world datasets.

- **Status:** manuscript under submission
- **Scope note:** the spectral analysis covers quadratic models; nonlinear
  experiments use full-batch gradients. Stochastic settings are not covered.

## Repository structure

```
.
├── lis_core.py                           # canonical LIS-GD implementation (shared)
├── lis_synthetic_regression.py           # Sec. 3.2, Table 1 - controlled synthetic regression
├── lis_regression_10seed.py              # Sec. 4.1, Table 3 - Advertising, Medical Insurance, California Housing (10 splits)
├── lis_advertising.py                    # Advertising, single split (seed 42)
├── lis_medical_insurance.py              # Medical Insurance, single split (seed 42)
├── lis_california_housing.py             # California Housing, single split (seed 42)
├── lis_diabetes_energy_10seed.py         # Sec. 4.1, Table 3 - Diabetes, Energy Efficiency (10 splits)
├── lis_classification_plateau_10seed.py  # Sec. 4.1, Table 3 - Wine, Digits, Breast Cancer (10 splits)
├── lis_tablo4_dbeta.py                   # Sec. 4.1, Table 4 - scale sensitivity and whitening (d_beta criterion)
├── lis_criterion_check.py                # Delta-L vs d_beta stopping-criterion comparison
├── mnist_full_comparison_gpu.py          # Sec. 4.2, Table 5 - MNIST, 8 BGD learning rates + LIS-GD (Colab GPU)
├── LISGD_fullGRID.py                     # Sec. 4.2, Table 6 - Fashion-MNIST, seven methods, budget-matched (Colab GPU)
├── mnist_seed_lr_grid.py                 # MNIST, 500-step consistency check (NumPy, Colab)
├── mnist_bgd_lisgd_m502.py               # MNIST, single-seed BGD vs LIS-GD (M=502) (NumPy, Colab)
├── results/                              # raw console output of the runs
└── exploratory/                          # earlier versions and development scripts, NOT used for the paper tables
```

## Which script produces which paper result

| Paper | Content | Script |
|---|---|---|
| Sec. 3.2, Table 1 | Synthetic linear regression (seed 42) | `lis_synthetic_regression.py` |
| Table 3 | Advertising, Medical Insurance, California Housing (d_beta criterion, 10 splits) | `lis_regression_10seed.py` |
| Table 3 | Diabetes, Energy Efficiency (M* minimizing training loss, iso-loss savings, 10 splits) | `lis_diabetes_energy_10seed.py` |
| Table 3 | Wine, Digits, Breast Cancer (training-accuracy plateau, 10 splits) | `lis_classification_plateau_10seed.py` |
| Table 4 | Superconductivity, Bike Sharing, House Prices, raw and whitened (d_beta criterion) | `lis_tablo4_dbeta.py` |
| Sec. 3 / 4 | Delta-L vs d_beta criterion comparison | `lis_criterion_check.py` |
| Table 5 | MNIST, best BGD vs LIS-GD, budgets 500-10000, M = budget + 2 | `mnist_full_comparison_gpu.py` |
| Table 6 | Fashion-MNIST, seven methods, budgets 500-5000, M = budget + 2 | `LISGD_fullGRID.py` |

**Stopping criterion.** Section 3 (Table 1) stops on the loss gap
(Delta-L <= 1e-8). For the three regression rows of Table 3 (Advertising,
Medical Insurance, California Housing) the criterion is the parameter distance
to the OLS solution, checked as the maximum absolute coordinate difference,
`max_i |theta_i - theta_OLS,i| <= 1e-8` (`lis_regression_10seed.py`). Table 4
and `lis_criterion_check.py` use the Euclidean (L2) distance
`||theta - theta_OLS||_2 <= 1e-8`. Raw output of the 10-split run is in
`results/regression_10seed_output.txt`.

The MNIST and Fashion-MNIST scripts were run on Google Colab (GPU). Their
console output is stored in `results/` so the reported numbers can be compared
without re-running. Table numbers follow the main text of the paper;
supplementary-material tables are described in each script's docstring.

Everything under `exploratory/` (earlier MNIST/Fashion-MNIST configurations,
the single-split Diabetes / Energy Efficiency scripts, the Delta-L whitening scan, the earlier classification script that selected M
on test accuracy, CIFAR-10 mini-batch runs, the condition-number sweep, the
Cycle-LIS ablation) was used during development but is **not** used for the
paper's tables. It is kept for transparency.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn` for the CPU scripts
- `torch`, `torchvision` for the neural-network scripts

```
pip install -r requirements.txt
```

## Data

Wine, Digits, Breast Cancer, Diabetes, California Housing, MNIST and
Fashion-MNIST are loaded automatically through `scikit-learn` / `torchvision`.

The following datasets must be downloaded by hand and placed as described in
each script's docstring:

| Dataset | Expected file | Source |
|---|---|---|
| Advertising | `Advertising.csv` | ISLR dataset |
| Medical Insurance | `insurance.csv` | "Medical Cost Personal Datasets" (Kaggle) |
| Energy Efficiency | `ENB2012_data.csv` | [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| Superconductivity | see `lis_whitened_mmin_scan.py` | [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data) |
| Bike Sharing | `day.csv` | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) |
| House Prices | `data/house_prices_train.csv` | [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (login required) |

## Reproducing a result

Scripts are self-contained. Run them from the repository root:

```
python lis_regression_10seed.py
```

## Citation

```bibtex
@unpublished{lisgd2026,
  title  = {LIS-GD: Finite-Horizon Gradient Optimization},
  author = {Kandemi{\c{s}}, U{\u{g}}ur and Candemir, Cemre},
  year   = {2026},
  note   = {Manuscript under submission}
}
```

A machine-readable version is in `CITATION.cff`. The archived release that
matches the paper is on Zenodo (DOI to be added with the release).
