# Search Without Recall – Algorithms Repository

This repository provides reference implementations of the algorithms in the appendix of the working paper *“Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies.”* The goal is to make **Algorithms 1–4** easy to run and verify; additional functions are supportive (interpolation, convergence checks, etc.).

> Double-blind note: author names are intentionally omitted.

---

## Repository Structure

```
.
├── algorithm.py          # Implementations of Algorithms 1–4 + helpers
├── README.md             # This document
├── LICENSE               # MIT License (standalone)
├── pyproject.toml        # Build + tooling (PEP 621), optional but recommended
├── requirements.txt      # Minimal dependency pinning for pip users
├── .gitignore            # Python, build, venv, cache
├── examples/
│   ├── quick_start.py          # Minimal runnable demos
│   └── reproduce_figure1.py    # Figure 1 at the reported resolution
└── tests/
    └── test_import.py    # Smoke test: imports and a short run
```

---

## Algorithms (paper appendix alignment)

* **Algorithm 1 — Scaled foresight recursion**
  Functions: `S_fast`, `_S_fast_compute_delta1`, `_S_fast_compute_delta_less1`, `S_c0`
  Computes $S_{n,k}$ over $(\mu, c)$ for $0<\delta<1$ or over $c$ for $\delta=1$.

* **Algorithm 2 — Maximum-horizon table**
  Function: `data_n_min`
  Produces the lookup table $k \mapsto n_{\max}(k)$ by scanning horizons.

* **Algorithm 3 — Thresholds**
  Function: `threshold`
  Locates the offer-value thresholds $\xi_k$ (accept/continue), including
  sign-changing, exact-grid, and tangent roots.

* **Algorithm 4 — Infinite horizon & critical discount factors**
  Functions: `S_c0_infinite`, `S_infinite`, `k_doublestar`, `delta_star`, `delta_k_star`
  Computes infinite-horizon limits $S_k$, the index $k^{**}$, and critical $\delta$.

* **Support routines**
  `get_S_value`, `get_S0_value` for safe interpolation/value extraction.

---

## Installation

### Option A — pip (simple)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — pyproject (recommended)

If you use a modern build tool (pip ≥23.1 supports PEP 517/518):

```bash
pip install .
```

This will read `pyproject.toml` and install dependencies and the package in editable mode (if configured).

> **Numba is optional**; if present, the code JIT-accelerates automatically.

---

## Quick Start

```python
from algorithm import PAPER_GRID_SIZE, S_fast

alpha0, nu0 = -0.5, 0.0
mu_flag, sigma_flag = 0, 0
beta, G = 1.0, 40  # quick demonstration; use PAPER_GRID_SIZE for reported results
n, delta = 100, 0.9

S_df, c_grid, mu_grid = S_fast(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)
print(S_df.head())
```

For more examples, see `examples/quick_start.py`.

To reproduce Figure 1, install the optional plotting dependency and run:

```bash
pip install -e '.[plots]'
python examples/reproduce_figure1.py /path/to/figure_1.png
```

---

## Testing

```bash
pip install -e '.[dev]'
pytest -q
# Or minimal smoke test
python -c "import algorithm; print('ok')"
```

---

## Reproducibility Notes

* The default resolution is `G=400`, exposed as `PAPER_GRID_SIZE`. The paper's
  companion identifies entries refined with an independent, finer-grid solver;
  those entries need not be reproduced at the default resolution.
* The state grids are deterministic given `(G, rho, Z, ita)`. The predictive
  innovation is integrated over the full real line using Gauss--Legendre
  quadrature after the transformation `u = tan(pi*z/2)`.
* Numba may change runtime but not results.
* For very high $\delta$ the code doubles the finite truncation horizon until
  the requested stages converge in sup norm; see the Algorithm 4 functions.
* Table labels retain full-precision grid coordinates. Use `get_S_value` or
  `get_S0_value` rather than formatting labels by hand. Previously saved tables
  with duplicate rounded labels must be recomputed. Both helpers interpolate
  linearly, using the adjacent end interval for extrapolation outside a grid.
* `data_n_min` reports a finite `n_max` only after finding a crossing. Otherwise
  `n_max` is `NaN`; `status` and `n_checked` distinguish the horizon cap,
  numerical stabilization, and excluded stages. Set `convergence_tol=0` to
  disable the stabilization shortcut. A small successive-horizon difference
  is not a bound on the remaining infinite-horizon error.
* `k_doublestar` returns `None` if no crossing is found through `max_k`.
  Discount searches raise an error if a required value computation fails.

---

## Citation (double-blind)

If you use this code, please cite:

> *Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies* (authors blinded).
> See appendix for Algorithms 1–4.

---

## License

Released under the **MIT License**. See `LICENSE`.
