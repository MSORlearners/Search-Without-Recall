# Search Without Recall – Algorithms Repository

This repository provides reference implementations of the algorithms in the appendix of the working paper *“Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies.”* The goal is to make **Algorithms 1–4** easy to run and verify; additional functions are supportive (interpolation, convergence checks, etc.).

> Double-blind note: author names are intentionally omitted.

---

## Repository Structure

```
.
├── algorithm.py          # Implementations of Algorithms 1–4 + helpers (annotated, no logic changes)
├── README.md             # This document
├── LICENSE               # MIT License (standalone)
├── pyproject.toml        # Build + tooling (PEP 621), optional but recommended
├── requirements.txt      # Minimal dependency pinning for pip users
├── .gitignore            # Python, build, venv, cache
├── examples/
│   └── quick_start.py    # Minimal runnable demo (Algorithm 1)
└── tests/
    └── test_import.py    # Smoke test: imports and a short run
```

---

## Algorithms (paper appendix alignment)

* **Algorithm 1 — Scaled foresight recursion**
  Functions: `S_fast`, `_S_fast_compute_delta1`, `_S_fast_compute_delta_less1`, `S_c0`
  Computes $S_{n,k}$ over $(\mu, c)$ for $0<\delta<1$ or over $c$ for $\delta=1$.

* **Algorithm 2 — Minimum horizon table**
  Function: `data_n_min`
  Produces the table $k \mapsto n^{*}(k)$ by scanning horizons.

* **Algorithm 3 — Thresholds**
  Function: `threshold`
  Locates decision thresholds $\xi_k$ (accept/continue) on the standardized grid.

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
from algorithm import S_fast

alpha0, nu0 = -0.5, 0.0
mu_flag, sigma_flag = 0, 0
beta, G = 1.0, 100
n, delta = 100, 0.9

S_df, c_grid, mu_grid = S_fast(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)
print(S_df.head())
```

For more examples, see `examples/quick_start.py`.

---

## Testing

```bash
# With pyproject: (if pytest listed)
pytest -q
# Or minimal smoke test
python -c "import algorithm; print('ok')"
```

---

## Reproducibility Notes

* The grids are deterministic given `(G, rho, Z, ita)`.
* Numba may change runtime but not results.
* For very high $\delta$ the code increases $n$ adaptively; see the Algorithm 4 functions for stopping criteria.

---

## Citation (double-blind)

If you use this code, please cite:

> *Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies* (authors blinded).
> See appendix for Algorithms 1–4.

---

## License

Released under the **MIT License**. See `LICENSE`.
