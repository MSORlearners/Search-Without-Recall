# Search Without Recall & Gaussian Learning — Algorithms

This repository implements the algorithms from the appendix of the working paper **“Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies.”**  
The purpose of this repository is to provide reproducible implementations of **Algorithms 1–4** listed in the paper’s appendix, with additional helper functions for posterior updates, interpolation, and convergence checks.

> Double-blind note: author names are intentionally omitted.

---

## Algorithms

- **Algorithm 1 – Scaled foresight recursion**  
  Backward recursion for the scaled foresight \( S_{n,k} \) on non-uniform grids, supporting both \(\delta=1\) and \(0<\delta<1\).

- **Algorithm 2 – Critical sample size \(k^\*\)**  
  Iteratively finds the largest horizon \(n\) for which very large offers are still rejected, giving \(k^\*(n,\delta)\).

- **Algorithm 3 – Accept/continue thresholds**  
  Recovers thresholds \((\xi^\ell_k,\xi^h_k)\) via grid search over standardized space \( \hat x \in [-\tau_k,\tau_k] \).

- **Algorithm 4 – Infinite-horizon limit**  
  Computes \( S_k \) and \( k^{\*\*}(\delta) = \lim_{n \to \infty} k^\*(n,\delta) \) via convergence in \(n\).

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U numpy scipy pandas numba
````

*Numba is optional; the code detects it and JIT-accelerates if available.*

---

## Quick start

```python
from algorithm import (
    compute_scaled_foresight,      # Algorithm 1
    compute_kstar_lookup,          # Algorithm 2
    compute_thresholds,            # Algorithm 3
    compute_infinite_horizon       # Algorithm 4
)

alpha0, nu0, delta, beta, G, n = -0.5, 0.0, 0.99, 1.0, 200, 100

# Algorithm 1
S, grids = compute_scaled_foresight(mu_flag=0, sigma_flag=0,
                                    alpha0=alpha0, nu0=nu0,
                                    n=n, delta=delta, beta=beta, G=G)

# Algorithm 2
kstar = compute_kstar_lookup(alpha0, nu0, delta, beta, G, k_upper=14)

# Algorithm 3
thresholds = compute_thresholds(k=6, mu_prev=0.0, sigma_prev=1.0,
                                alpha0=alpha0, nu0=nu0,
                                delta=delta, c=0.0, S_df=S, G=G)

# Algorithm 4
S_inf, kss = compute_infinite_horizon(alpha0, nu0, delta, beta, G, k_upper=14)
```

---

## Repository structure

```
algorithm.py   # implementations of Algorithms 1–4 and helper routines
README.md      # project documentation
```

---

## Citation (double-blind)

If you use this code, please cite:

> *Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies* (authors blinded).
> Algorithms appear in the appendix: Algorithm 1 (scaled foresight), Algorithm 2 (critical sample size), Algorithm 3 (thresholds), Algorithm 4 (infinite horizon).

---

## License

**MIT License**

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
