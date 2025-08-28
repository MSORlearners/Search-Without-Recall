# Search Without Recall – Algorithms Repository

This repository provides reference implementations of the algorithms described in the appendix of the working paper *“Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies”*.

The code is written in Python and makes heavy use of Numba for acceleration. Each algorithm in the paper corresponds to a callable function in `algorithm.py`, with additional support routines for interpolation, convergence checks, and infinite-horizon limits.

---

## Repository Structure

```
.
├── algorithm.py     # Core implementations (Algorithms 1–4 + helpers)
├── README.md        # This file
```

---

## Algorithms

### Algorithm 1 — Scaled foresight recursion

* **Functions**: `S_fast`, `_S_fast_compute_delta1`, `_S_fast_compute_delta_less1`, `S_c0`
* Implements the backward recursion for finite horizon $n$, computing the scaled foresight function $S_{n,k}$ over $(μ,c)$ when $δ<1$, or over $c$ alone when $δ=1$.

### Algorithm 2 — Minimum horizon table

* **Function**: `data_n_min`
* Computes the smallest horizon $n$ at which acceptance occurs for each $k$. Produces a table of $k \mapsto n^*(k)$.

### Algorithm 3 — Thresholds

* **Function**: `threshold`
* Locates threshold points $ξ_k$ that separate “accept” and “continue” regions. Works both in finite-horizon and infinite-horizon settings.

### Algorithm 4 — Infinite horizon and critical discount factors

* **Functions**: `S_c0_infinite`, `S_infinite`, `k_doublestar`, `delta_star`, `delta_k_star`
* Computes infinite-horizon limits of $S$, identifies the $k^{**}$ index, and locates critical discount factors $δ^*$ at which policy shifts occur.

### Support routines

* **Functions**: `get_S_value`, `get_S0_value`
* Provide interpolation and value extraction from computed matrices.
* Used internally by Algorithms 2–4.

---

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# (Optional) create and activate a virtual environment

# Install dependencies
pip install -r requirements.txt
```

Minimal usage example:

```python
from algorithm import S_fast

# Parameters
alpha0, nu0, mu, sigma, beta, G = -0.5, 0, 0, 0, 1, 100
n, delta = 100, 0.9

# Run Algorithm 1
S_df, c_grid, mu_grid = S_fast(mu, sigma, alpha0, nu0, n, delta, beta, G)
print(S_df.head())
```

---

## Citation

If you use this code in your research, please cite the working paper:

```
@misc{search_gaussian_learning,
  title   = {Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies},
  note    = {Working paper},
  year    = {2025}
}
```

---

## License

This repository is released under the MIT License. See `LICENSE` for details.
