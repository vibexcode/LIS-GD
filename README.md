# LIS-GD: Finite-Horizon Gradient Optimization

This repository contains the code and experiments accompanying the paper
**"LIS-GD: Finite-Horizon Gradient Optimization"** ("LIS-GD: Sonlu Ufuklu
Gradyan Optimizasyonu").

LIS-GD is a gradient-descent method whose per-step coefficients are derived
in closed form from a single horizon parameter, **M**, via the *Logarithmic
Information Surplus* (LIS) profile:

```
I_extra(k; M) = ln(k) / ln(M) - k / M,     k = 2, ..., M
theta_{n+1}   = theta_n - I_extra(n+2; M) * grad L(theta_n),   n = 0, ..., M-2
```

The coefficient rises, peaks once, and returns to exactly zero at k = M,
producing a self-terminating optimizer with no separate stopping rule and
no external learning-rate schedule. The paper establishes this geometry
analytically (Sections 2-3) and evaluates it empirically across thirteen
real-world datasets and one controlled synthetic experiment (Section 4).

- **Status:** preprint (update once a venue decision is made)
- **Preprint / PDF:** _add link once available_

## Repository structure

```
.
├── lis_core.py                       # shared canonical LIS-GD implementation
├── lis_synthetic_regression.py       # Sec. 3.2 / Ek A.1 — controlled synthetic regression
├── lis_advertising.py                # Sec. 4.1 — Advertising
├── lis_medical_insurance.py          # Sec. 4.1 — Medical Insurance
├── lis_california_housing.py         # Sec. 4.1 — California Housing
├── lis_diabetes.py                   # Sec. 4.2 — Diabetes
├── lis_energy_efficiency.py          # Sec. 4.2 — Energy Efficiency
├── lis_classification_suite.py       # Sec. 4.3 — Wine, Digits, Breast Cancer
├── lis_whitened_mmin_scan.py         # Sec. 4.4 — Superconductivity, Bike Sharing, House Prices
├── lis_baseline_comparison.py        # Sec. 4.5 — MNIST (numpy)
├── lis_baseline_comparison_torch.py  # Sec. 4.5 — MNIST (PyTorch/GPU)
├── lis_baseline_fashion_mnist.py     # Sec. 4.5 — Fashion-MNIST, single seed
├── lis_multiseed_fashion_mnist.py    # Sec. 4.5, Table 13 — Fashion-MNIST, 5-seed comparison
└── exploratory/                      # NOT cited in the current paper (see below)
    ├── lis_correlation_sweep.py
    ├── lis_cifar10_minibatch.py
    └── lis_cycle_baseline_fashion_mnist.py
```

> **Note:** if your local layout still has these scripts under `tests/`
> instead of the repository root, either move them to match the tree above
> or update the paths in this README to match — keep the two consistent.

## Which script produces which paper result

All thirteen real datasets in Section 4, plus the Section 3.2 synthetic
experiment, are covered:

| Paper section | Dataset(s) | Script |
|---|---|---|
| 3.2 / Ek A.1 | Synthetic linear regression (seed=42) | `lis_synthetic_regression.py` |
| 4.1 | Advertising | `lis_advertising.py` |
| 4.1 | Medical Insurance | `lis_medical_insurance.py` |
| 4.1 | California Housing | `lis_california_housing.py` |
| 4.2 | Diabetes | `lis_diabetes.py` |
| 4.2 | Energy Efficiency | `lis_energy_efficiency.py` |
| 4.3 | Wine, Digits, Breast Cancer | `lis_classification_suite.py` |
| 4.4 | Superconductivity, Bike Sharing, House Prices | `lis_whitened_mmin_scan.py` |
| 4.5, Table 12 | MNIST | `lis_baseline_comparison.py` / `lis_baseline_comparison_torch.py` |
| 4.5, Table 13 | Fashion-MNIST (multi-seed) | `lis_multiseed_fashion_mnist.py` |

Everything under `exploratory/` (CIFAR-10 mini-batch runs, the
condition-number correlation sweep, the Cycle-LIS ablation on
Fashion-MNIST) was used to develop and stress-test the theory during
writing but is **not** referenced by the current (23-page) version of the
paper. It's kept for transparency, not as a reproduction target.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn` — all CPU-based scripts
- `torch`, `torchvision` — the GPU-based neural-network scripts
  (`lis_baseline_comparison_torch.py`, `lis_baseline_fashion_mnist.py`,
  `lis_multiseed_fashion_mnist.py`)

```
pip install numpy pandas scikit-learn torch torchvision
```

Some scripts expect a dataset CSV in the same directory (see each script's
own docstring for the exact filename and source — e.g. `Advertising.csv`,
`insurance.csv`, `ENB2012_data.csv`); others (Diabetes, Wine, Digits,
Breast Cancer, MNIST, Fashion-MNIST) fetch or load their data automatically
via `sklearn`/`torchvision`.

## Reproducing a result

Each script is self-contained and runnable directly:

```
python lis_diabetes.py
```

It prints the LIS-GD/BGD comparison and the specific paper numbers it's
meant to reproduce, so you can check the two side by side without cross-
referencing the PDF.

## Citation

If you use this code or refer to this work, please cite:

```bibtex
@unpublished{lisgd2026,
  title  = {LIS-GD: Finite-Horizon Gradient Optimization},
  author = {<Author Name>},
  year   = {2026},
  note   = {Preprint}
}
```

(Update the author field and note once the paper has an official
citation/DOI.)
