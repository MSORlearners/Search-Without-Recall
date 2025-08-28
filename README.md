# Search Without Recall and Gaussian Learning (SWoR-GL)

Efficient, numerically stable dynamic-programming (DP) routines for **Search Without Recall** under **Gaussian learning**, accompanying the working paper:

> **Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies**  

This repository provides:
- Fast DP recursions for value indices \(S(k,\mu,c)\) and the special case \(S_0(k,\mu)=S(k,\mu,0)\).
- Infinite-horizon approximations and critical-threshold utilities (`delta_star`, `delta_k_star`, `k_doublestar`).
- Optional JIT acceleration via **Numba**.

## Features
- **Stable densities**: log-space Student-t PDF, safe Normal PDF.
- **Nonuniform grids** for robust interpolation near boundaries.
- **Vectorized / parallel loops** with optional `numba.njit(parallel=True)`.
- **Delta = 1** and **Delta < 1** branches with consistent APIs.

## Installation
```bash
pip install -e .
