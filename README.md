
## Paper

- **Title:** LIS-GD: Sonlu Ufuklu Gradyan Optimizasyonu
- **Status:** [submitted / under review / preprint — update as appropriate]
- **PDF / preprint link:** [add link once available]

## Experiments

All code used to produce the paper's empirical results (Section 4) and
supporting exploratory analyses is in [`tests/`](./tests). This
includes real-world dataset comparisons (MNIST, Fashion-MNIST,
California Housing, Breast Cancer, Wine, Digits, Superconductivity,
Bike Sharing, House Prices), baseline comparisons against tuned
constant learning rates, standard schedules (linear decay, cosine
annealing, triangular CLR, one-cycle), and AdamW, and a whitening/
conditioning study tied to the paper's theoretical results. See
[`tests/README.md`](./tests/README.md) for a script-by-script
description and instructions to reproduce each result.

## Requirements

- Python 3.9+
- `numpy`, `scikit-learn` for all CPU-based scripts
- `torch`, `torchvision` for the GPU-based neural-network scripts
- See [`tests/README.md`](./tests/README.md) for per-script details

## Citation

If you use this code or refer to this work, please cite:
