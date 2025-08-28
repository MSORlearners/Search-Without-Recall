import math
import numpy as np
import pandas as pd
from scipy.stats import t as _scipy_t, norm as _scipy_norm  # (not used in numba paths)
from functools import lru_cache

# Optional acceleration
try:
    import numba
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

"""
Algorithms for Search Without Recall with Gaussian Learning
----------------------------------------------------------

This module implements the dynamic-programming (DP) value recursions described in the
working paper:

  "Search Without Recall and Gaussian Learning: Structural Properties and Optimal Policies"
   (Xu et al., 2025; working paper).

Core Objects
------------
- S(k, μ, c): Value index for state (k, μ, c) under discount δ ∈ (0,1].
- Special case S0(k, μ) := S(k, μ, 0).

Key Routines
------------
- S_fast(...)         : Finite-horizon backward recursion for S; supports δ=1 and δ<1.
- S_c0(...)           : Finite-horizon recursion restricted to c=0 (S0).
- S_infinite(...)     : Fixed-point style convergence for the general S (increasing horizon n).
- S_c0_infinite(...)  : As above for S0.
- Interpolation helpers: get_S_value, get_S0_value.
- Threshold/Phase utilities: threshold, delta_star, delta_k_star, k_doublestar, data_n_min.

Numerical Notes
---------------
1) We evaluate Student-t densities in log-space for stability.
2) Grids are non-uniform to concentrate points near boundaries (±τ).
3) Interpolation is linear/bilinear over the constructed grids.
4) Numba accelerates the tight loops when available; logic falls back to NumPy otherwise.

Parameters (conventions)
------------------------
- mu_flag ∈ {0,1}, sigma_flag ∈ {0,1} toggle analytical branches used in the paper.
- (alpha0, nu0) parameterize prior strength (df and location scaling).
- delta ∈ (0,1] is the discount factor; beta is a scale (commonly 1).
- G controls grid resolution.

DISCLAIMER: This code is research software accompanying a working paper. Interface and
implementations may change; please verify results for your use case.
"""


# --------------------------- Density helpers (Numba-friendly) ---------------------------

if _HAS_NUMBA:
    @njit
    def _t_pdf(x, df):
        """
        Student-t PDF in log-space for numerical stability.
        """
        if df <= 0:
            return 0.0
        log_term1 = math.lgamma((df + 1) / 2) - (0.5 * math.log(df * math.pi) + math.lgamma(df / 2))
        log_term2 = -((df + 1) / 2) * math.log1p((x * x) / df)
        return math.exp(log_term1 + log_term2)

    @njit
    def _norm_pdf(x):
        """
        Standard Normal PDF.
        """
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @njit
    def _make_grids(G, rho=0.85, Z=30.0, ita=0.75):
        """
        Construct descending c-grid and symmetric z-grid; also returns μ-grid (= z-grid),
        and centered finite-difference spacings dz for integration.

        The cubic blend (ita) yields denser coverage near the extremes for improved stability.
        """
        c       = np.empty(G+2, dtype=np.float64)
        z       = np.empty(G+2, dtype=np.float64)
        mu_grid = np.empty(G+2, dtype=np.float64)
        for j in range(G+2):
            c[j]       = G * rho**j
            z[j]       = Z * ((1-ita)*(2*j-G-1)/(G-1) + ita*((2*j-G-1)/(G-1))**3)
            mu_grid[j] = z[j]
        c[G] = 0.0
        dz = np.empty_like(z)
        dz[0]  = z[1]  - z[0]
        dz[-1] = z[-1] - z[-2]
        for j in range(1, G+1):
            dz[j] = 0.5 * (z[j+1] - z[j-1])
        return c, z, mu_grid, dz

else:
    # CPU (non-numba) fallbacks used rarely (kept minimal)
    def _t_pdf(x, df):
        return _scipy_t.pdf(x, df) if df > 0 else 0.0

    def _norm_pdf(x):
        return _scipy_norm.pdf(x)

    def _make_grids(G, rho=0.85, Z=30.0, ita=0.75):
        c       = np.array([G * rho**j for j in range(G+2)], dtype=float)
        z_raw   = np.array([Z * ((1-ita)*(2*j-G-1)/(G-1) + ita*((2*j-G-1)/(G-1))**3) for j in range(G+2)], dtype=float)
        mu_grid = z_raw.copy()
        c[G]    = 0.0
        dz      = np.empty_like(z_raw)
        dz[0]   = z_raw[1] - z_raw[0]
        dz[-1]  = z_raw[-1] - z_raw[-2]
        for j in range(1, G+1):
            dz[j] = 0.5 * (z_raw[j+1] - z_raw[j-1])
        return c, z_raw, mu_grid, dz


# ------------------------------- Core DP kernels (Numba) --------------------------------

if _HAS_NUMBA:

    @njit(parallel=True)
    def _S_fast_compute_delta1(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
        """
        Backward recursion for S when delta == 1.
        Returns (S_arr, k_min, c, mu_grid), with S_arr shape (n, G+2).
        """
        c, z, mu_grid, dz = _make_grids(G)

        if sigma_flag == 0:
            k_min = max(math.floor(2 - 2*alpha0), 1)
        else:
            k_min = 1

        n_int = int(n)
        G_int = int(G)
        S_arr = np.zeros((n_int, G_int+2), dtype=np.float64)

        for k in range(n_int-1, k_min-1, -1):
            df = 2*alpha0 + k
            for j_c in numba.prange(1, G_int+1):
                if k < n_int-1:
                    acc = 0.0
                    for j_u in numba.prange(1, G_int+1):
                        mu_u = 0.0 if mu_flag==1 else 1.0/(nu0+k+1)
                        if sigma_flag == 0:
                            L = math.sqrt(max(0.0, (1-mu_u*mu_u)/(df+1)))
                            s = L * math.sqrt(df + z[j_u]*z[j_u])
                            pdf_val = _t_pdf(z[j_u], df)
                        else:
                            s = math.sqrt(1-mu_u*mu_u)
                            pdf_val = _norm_pdf(z[j_u])

                        x = c[j_c]/s
                        i_c = G_int
                        while i_c > 1 and x > c[i_c]:
                            i_c -= 1
                        if i_c == G_int:
                            i_c = G_int-1
                        frac = (x - c[i_c])/(c[i_c+1]-c[i_c])
                        S_u = (1-frac)*S_arr[k+1,i_c] + frac*S_arr[k+1,i_c+1]

                        val = max(0.0, (-1+delta*mu_u)*z[j_u] + s*S_u - c[j_c])
                        acc += val * pdf_val * dz[j_u]
                    S_arr[k,j_c] = beta * delta * acc
        return S_arr, k_min, c, mu_grid

    @njit(parallel=True)
    def _S_fast_compute_delta_less1(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
        """
        Backward recursion for S when delta < 1.
        Returns (S_arr, k_min, c, mu_grid), with S_arr shape (n, G+2, G+2).
        """
        c, z, mu_grid, dz = _make_grids(G)

        if sigma_flag == 0:
            k_min = max(math.floor(2 - 2*alpha0), 1)
        else:
            k_min = 1

        n_int = int(n)
        G_int = int(G)
        S_arr = np.zeros((n_int, G_int+2, G_int+2), dtype=np.float64)

        for k in range(n_int-1, k_min-1, -1):
            df = 2*alpha0 + k
            for j_c in numba.prange(1, G_int+1):
                for j_mu in numba.prange(1, G_int+1):
                    if k < n_int-1:
                        acc = 0.0
                        mu0 = mu_grid[j_mu]
                        for j_u in numba.prange(1, G_int+1):
                            mu_u = z[j_u]/(nu0+k+1)
                            if sigma_flag == 0:
                                L = math.sqrt(max(0.0,(1-(1/(nu0+k+1))**2)/(df+1)))
                                s = L * math.sqrt(df + z[j_u]*z[j_u])
                                pdf_val = _t_pdf(z[j_u], df)
                            else:
                                s = math.sqrt(1-(1/(nu0+k+1))**2)
                                pdf_val = _norm_pdf(z[j_u])

                            new_mu = (mu0 + mu_u)/s
                            new_c  = c[j_c]/s

                            i_mu = G_int
                            while i_mu > 1 and new_mu < mu_grid[i_mu]:
                                i_mu -= 1
                            if i_mu == G_int:
                                i_mu = G_int-1
                            frac_mu = (new_mu - mu_grid[i_mu])/(mu_grid[i_mu+1]-mu_grid[i_mu])

                            i_c = G_int
                            while i_c > 1 and new_c > c[i_c]:
                                i_c -= 1
                            if i_c == G_int:
                                i_c = G_int-1
                            frac_c = (new_c - c[i_c])/(c[i_c+1]-c[i_c])

                            S_interp = (
                                (1-frac_mu)*(1-frac_c)*S_arr[k+1, i_mu,   i_c  ] +
                                (  frac_mu)*(1-frac_c)*S_arr[k+1, i_mu+1, i_c  ] +
                                (1-frac_mu)*(  frac_c)*S_arr[k+1, i_mu,   i_c+1] +
                                (  frac_mu)*(  frac_c)*S_arr[k+1, i_mu+1, i_c+1]
                            )

                            integrand = max(
                                0.0,
                                -(1 - delta/(nu0 + k + 1)) * z[j_u]
                                - (1 - delta)*mu0
                                + s*S_interp
                                - c[j_c]
                            )
                            acc += integrand * pdf_val * dz[j_u]
                        S_arr[k, j_mu, j_c] = delta * acc
        return S_arr, k_min, c, mu_grid


# ------------------------------ Public DP wrappers (API) ------------------------------

def _S_fast_compute(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Dispatch to the correct kernel based on delta.
    """
    if abs(delta - 1.0) < 1e-10:
        if not _HAS_NUMBA:
            raise RuntimeError("delta==1 path requires numba in this implementation.")
        return _S_fast_compute_delta1(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)
    else:
        if not _HAS_NUMBA:
            raise RuntimeError("delta<1 path requires numba in this implementation.")
        return _S_fast_compute_delta_less1(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)


def S_fast(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Finite-horizon DP for S(k, μ, c) (or S(k, c) when δ=1).

    Returns
    -------
    (df, c_grid, mu_grid)
        - If δ=1: df has rows indexed by 'n=..., k_min=..., k=...' and columns 'c=...'.
        - If δ<1: df has rows 'k=..., c=...' and columns 'mu=...'.
    """
    S_arr, k_min, c, mu_grid = _S_fast_compute(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)

    if abs(delta - 1.0) < 1e-10:
        df = pd.DataFrame(S_arr[k_min:n, 1:G+1])
        df.columns = [f'c={c[j]:.1f}' for j in range(1, G+1)]
        df.index   = [f'n={n}, k_min={k_min}, k={k}' for k in range(k_min, n)]
        return df, c, mu_grid

    rows, row_labels = [], []
    for k in range(k_min, n):
        for j_c in range(1, G+1):
            rows.append(S_arr[k, 1:G+1, j_c])
            row_labels.append(f'k={k}, c={c[j_c]:.6f}')
    df = pd.DataFrame(rows)
    df.columns = [f'mu={mu_grid[j]:.6f}' for j in range(1, G+1)]
    df.index   = row_labels
    return df, c, mu_grid


if _HAS_NUMBA:
    @njit(parallel=True)
    def _S_c0_compute(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
        """
        Backward recursion specialized to c=0 (S0).
        Returns (S_arr, k_min, mu_grid) with S_arr shape (n, G+2).
        """
        _, z, mu_grid, dz = _make_grids(G)

        if sigma_flag == 0:
            k_min = max(math.floor(2 - 2*alpha0), 1)
        else:
            k_min = 1

        n_int = int(n)
        G_int = int(G)
        S_arr = np.zeros((n_int, G_int+2), dtype=np.float64)

        for k in range(n_int-1, k_min-1, -1):
            df = 2*alpha0 + k

            if abs(delta - 1.0) < 1e-10:
                for _j_c in range(1, 2):
                    if k < n_int-1:
                        acc = 0.0
                        for j_u in numba.prange(1, G_int+1):
                            if mu_flag == 0:
                                mu_u = 1/(nu0+k+1)
                            else:
                                mu_u = 0.0
                            if sigma_flag == 0:
                                Lambda_k1 = math.sqrt(max(0.0, (1-(1/(nu0+k+1))**2)/(2*alpha0+k+1)))
                                s = Lambda_k1 * math.sqrt(2*alpha0 + k + z[j_u]*z[j_u])
                                pdf_val = _t_pdf(z[j_u], 2*alpha0+k)
                            else:
                                s = math.sqrt(1-(1/(nu0+k+1))**2)
                                pdf_val = _norm_pdf(z[j_u])

                            integrand = max(0.0, -(nu0 + k)/(nu0 + k + 1) * z[j_u] + s * S_arr[k+1, 1])
                            acc += integrand * pdf_val * dz[j_u]
                        S_arr[k, 1] = acc
            else:
                for j_mu in numba.prange(1, G_int+1):
                    if k < n_int-1:
                        acc = 0.0
                        mu0 = mu_grid[j_mu]
                        for j_u in numba.prange(1, G_int+1):
                            mu_u = z[j_u]/(nu0+k+1)
                            if sigma_flag == 0:
                                L = math.sqrt(max(0.0,(1-(1/(nu0+k+1))**2)/(df+1)))
                                s = L*math.sqrt(2*alpha0+k + z[j_u]*z[j_u])
                                pdf_val = _t_pdf(z[j_u], df)
                            else:
                                s = math.sqrt(1-(1/(nu0+k+1))**2)
                                pdf_val = _norm_pdf(z[j_u])

                            new_mu = (mu0 + mu_u)/s
                            i_mu = G_int
                            while i_mu > 1 and new_mu < mu_grid[i_mu]:
                                i_mu -= 1
                            if i_mu == G_int:
                                i_mu = G_int-1
                            frac_mu = (new_mu - mu_grid[i_mu])/(mu_grid[i_mu+1]-mu_grid[i_mu])
                            S_interp = (1-frac_mu)*S_arr[k+1, i_mu] + frac_mu*S_arr[k+1, i_mu+1]

                            integrand = max(0.0, -(1 - delta/(nu0 + k + 1)) * z[j_u] - (1 - delta)*mu0 + s*S_interp)
                            acc += integrand * pdf_val * dz[j_u]
                        S_arr[k, j_mu] = delta * acc
        return S_arr, k_min, mu_grid
else:
    def _S_c0_compute(*args, **kwargs):
        raise RuntimeError("S_c0 requires numba in this implementation.")


def S_c0(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Finite-horizon DP restricted to c=0 (S0).
    Returns
    -------
    (df, mu_grid)
        - If δ=1: df has a single column 'S0' with rows 'k=i'.
        - If δ<1: df has rows 'k=i' and columns 'mu=...'.
    """
    S_arr, k_min, mu_grid = _S_c0_compute(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)

    if abs(delta - 1.0) < 1e-10:
        df = pd.DataFrame(S_arr[k_min:n, 1:2], columns=['S0'])
        df.index = [f'k={k}' for k in range(k_min, n)]
        mu_grid = np.array([0.0])
        return df, mu_grid

    df = pd.DataFrame(S_arr[k_min:n, 1:G+1])
    df.columns = [f'mu={mu_grid[j]:.6f}' for j in range(1, G+1)]
    df.index   = [f'k={k}' for k in range(k_min, n)]
    mu_grid = mu_grid[1:G+1]
    return df, mu_grid


# ------------------------------ Interpolation helpers ------------------------------

def get_S0_value(S0_df_tuple, k, mu_val):
    """
    Interpolate S0(k, μ) from an S_c0(...) result.

    Parameters
    ----------
    S0_df_tuple : (DataFrame, mu_grid) | DataFrame
        Return of S_c0; for δ=1, mu is ignored and a single-column DF is expected.
    k : int
    mu_val : float | None
    """
    if isinstance(S0_df_tuple, tuple) and len(S0_df_tuple) == 2:
        S0_df, mu_grid = S0_df_tuple
    else:
        S0_df, mu_grid = S0_df_tuple, None

    is_delta_one = (len(S0_df.columns) == 1 and S0_df.columns[0] == 'S0')
    available_k = sorted(set(int(lbl.split('=')[1]) for lbl in S0_df.index))
    if k not in available_k:
        raise KeyError(f"k={k} not in {available_k}")

    if is_delta_one:
        return float(S0_df.loc[f'k={k}', 'S0'])

    if mu_val is None:
        raise ValueError("mu_val is required for δ<1 case.")

    # grid-based linear interpolation
    i_mu = len(mu_grid) - 1
    while i_mu > 1 and mu_val < mu_grid[i_mu]:
        i_mu -= 1
    i_mu = min(i_mu, len(mu_grid) - 2)
    mu_lo, mu_hi = mu_grid[i_mu], mu_grid[i_mu+1]
    w = 0.0 if mu_hi == mu_lo else (mu_val - mu_lo) / (mu_hi - mu_lo)

    s_lo = float(S0_df.loc[f'k={k}', f'mu={mu_lo:.6f}'])
    s_hi = float(S0_df.loc[f'k={k}', f'mu={mu_hi:.6f}'])
    return s_lo * (1 - w) + s_hi * w


def get_S_value(S_df, k, mu_val, c_val):
    """
    Bilinear interpolation for S(k, μ, c).

    - If δ<1: rows 'k=..., c=...' and μ-columns. Interpolates over (μ, c).
    - If δ=1 : rows 'n=..., k_min=..., k=...' and c-columns. Interpolates over c only.
    """
    if 'mu=' in S_df.columns[0]:  # δ<1
        available_k = sorted(set(int(idx.split('k=')[1].split(',')[0]) for idx in S_df.index))
        if k not in available_k:
            raise KeyError(f"k={k} not in {available_k}")
        if mu_val is None:
            raise ValueError("mu_val is required when δ<1")

        mu_cols = [float(c.split('=')[1]) for c in S_df.columns]
        mu_lo = max([m for m in mu_cols if m <= mu_val], default=mu_cols[0])
        mu_hi = min([m for m in mu_cols if m >= mu_val], default=mu_cols[-1])
        if mu_lo == mu_hi:
            mu_hi = mu_cols[min(mu_cols.index(mu_lo) + 1, len(mu_cols)-1)]

        c_vals = sorted(set(float(idx.split('c=')[1]) for idx in S_df.index if f'k={k}' in idx))
        c_lo = max([c for c in c_vals if c <= c_val], default=c_vals[0])
        c_hi = min([c for c in c_vals if c >= c_val], default=c_vals[-1])
        if c_lo == c_hi:
            c_hi = c_vals[min(c_vals.index(c_lo) + 1, len(c_vals)-1)]

        def lookup(kv, cv, mv):
            return float(S_df.loc[f'k={kv}, c={cv:.6f}', f'mu={mv:.6f}'])

        S_ll = lookup(k, c_lo, mu_lo)
        S_lu = lookup(k, c_lo, mu_hi)
        S_ul = lookup(k, c_hi, mu_lo)
        S_uu = lookup(k, c_hi, mu_hi)

        w_mu = 0.0 if mu_hi == mu_lo else (mu_val - mu_lo) / (mu_hi - mu_lo)
        w_c  = 0.0 if c_hi  == c_lo  else (c_val - c_lo)  / (c_hi  - c_lo)

        return (S_ll * (1 - w_mu) * (1 - w_c)
              + S_lu * w_mu * (1 - w_c)
              + S_ul * (1 - w_mu) * w_c
              + S_uu * w_mu * w_c)

    # δ=1
    rows_for_k = [idx for idx in S_df.index if f', k={k}' in idx]
    if not rows_for_k:
        raise KeyError(f"No row found for k={k}")
    row_idx = rows_for_k[0]
    c_cols = [float(c.split('=')[1]) for c in S_df.columns]
    c_lo = max([c for c in c_cols if c <= c_val], default=c_cols[0])
    c_hi = min([c for c in c_cols if c >= c_val], default=c_cols[-1])
    if c_lo == c_hi:
        c_hi = c_cols[min(c_cols.index(c_lo) + 1, len(c_cols)-1)]

    s_lo = float(S_df.loc[row_idx, f'c={c_lo:.1f}'])
    s_hi = float(S_df.loc[row_idx, f'c={c_hi:.1f}'])
    w_c = 0.0 if c_hi == c_lo else (c_val - c_lo) / (c_hi - c_lo)
    return s_lo * (1 - w_c) + s_hi * w_c


# --------------------------- Infinite-horizon & thresholds ---------------------------

def data_n_min(mu, sigma, alpha0, nu0, delta, beta, G, k_upper):
    """
    Scan minimal finite horizon n for each k up to k_upper where S0 exceeds τ.

    Notes
    -----
    - Uses S_c0(.) and get_S0_value(.) with a convergence guard.
    - For δ close to 1, raises n_upper to ensure stabilization.
    """
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1

    k_value = k_min
    n_min   = k_value + 2
    results = []

    n_upper = 3000 if delta > 0.998 else 500 if delta < 0.99 else 1000

    while k_value <= k_upper:
        try:
            if nu0 + k_value == 1:
                results.append([k_value, 'Null-Fully Responsive'])
            else:
                twoalpha = 2*alpha0 + k_value
                nu = nu0 + k_value
                Lambda = np.sqrt((1 - (1/nu)**2) / twoalpha)
                tau = (1 - delta*beta/nu)/Lambda
                mu_star = None if abs(delta-1.0) < 1e-10 else 1/((nu0 + k_value)*Lambda)

                S_0 = 0.0
                n = max(n_min, k_value + 2)
                bound = 0
                S0_prev = None

                while True:
                    S0_result = S_c0(mu, sigma, alpha0, nu0, n, delta, beta, G)
                    S_0 = float(get_S0_value(S0_result, k_value, None if mu_star is None else mu_star))

                    if abs(delta - 1.0) >= 1e-10 and S0_prev is not None:
                        if abs(S_0 - S0_prev) < 1e-6 or n >= n_upper+1:
                            bound = 1
                            break
                    S0_prev = S_0

                    if S_0 > tau:
                        break
                    n += 1

                if bound == 1:
                    for k in range(k_value, k_upper + 1):
                        results.append([k, 'Null'])
                    break

                n_min = n - 1
                results.append([k_value, n_min])

        except Exception as e:
            results.append([k_value, f'Null-Error: {e}'])

        k_value += 1

    present = {r[0] for r in results}
    for k in range(1, k_upper + 1):
        if k not in present:
            results.append([k, 'Null-Missing'])

    results.sort(key=lambda x: x[0])
    df = pd.DataFrame(results, columns=['k', 'n_min'])
    df.index.name = 'Index'
    return df


def S_c0_infinite(mu, sigma, alpha0, nu0, delta, beta, G):
    """
    Infinite-horizon approximation for S0 via horizon-doubling / stepping.

    Returns
    -------
    (S_df, mu_grid)
        Approximate fixed point (n large).
    """
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1

    if 0.999 <= delta < 1:
        n_initial = min(int(1/(1-delta)), 10000)
    elif 0.995 <= delta < 0.999:
        n_initial = 2000
    elif 0.990 <= delta < 0.995:
        n_initial = 1000
    else:
        n_initial = 500

    n = int(n_initial)
    S_n, mu_grid = S_c0(mu, sigma, alpha0, nu0, n, delta, beta, G)
    S_n1, _      = S_c0(mu, sigma, alpha0, nu0, n+1, delta, beta, G)

    converged = False
    iteration = 0
    while not converged and iteration < 10:
        iteration += 1
        s0  = float(get_S0_value(S_n,  k_min, 0 if abs(delta-1.0)<1e-10 else 0))
        s1  = float(get_S0_value(S_n1, k_min, 0 if abs(delta-1.0)<1e-10 else 0))
        diff = s1 - s0
        if diff < 1e-4:
            converged = True
        else:
            if n - n_initial > 2:
                n_initial *= 2
                n = int(n_initial)
                S_n, mu_grid = S_c0(mu, sigma, alpha0, nu0, n, delta, beta, G)
            else:
                n += 1
                S_n = S_n1
            S_n1, _ = S_c0(mu, sigma, alpha0, nu0, n+1, delta, beta, G)

    return S_n1, mu_grid


def S_infinite(mu, sigma, alpha0, nu0, delta, beta, G, k_upper):
    """
    Infinite-horizon approximation for general S via horizon stepping.

    Returns
    -------
    (S_df, mu_grid)
        Approximate fixed point (n large).
    """
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1

    n_initial = 1000
    n = n_initial

    S_n, c, mu_grid = S_fast(mu, sigma, alpha0, nu0, n,   delta, beta, G)
    S_n1, _, _      = S_fast(mu, sigma, alpha0, nu0, n+1, delta, beta, G)

    converged = False
    n_max = 1001

    while not converged and n < n_max:
        S_n_kmin  = S_n.filter(like=f'k={k_min},', axis=0)
        S_n1_kmin = S_n1.filter(like=f'k={k_min},', axis=0)
        if len(S_n_kmin) == 0 or len(S_n1_kmin) == 0:
            break
        diff = float(np.max(np.abs(S_n1_kmin.values - S_n_kmin.values)))
        if diff < 1e-6 or n >= n_max - 1:
            converged = True
        else:
            n += 1
            S_n = S_n1
            S_n1, _, _ = S_fast(mu, sigma, alpha0, nu0, n+1, delta, beta, G)

    return S_n1, mu_grid


# ------------------------------ Thresholds & critical deltas ------------------------------

def threshold(k, mu_prev, sigma_prev, alpha0, nu0, delta, S_df, C):
    """
    Find roots x in [-τ, τ] of f(x) = S(k, μ, c) - c - x - (1-δ)μ
    using a dense edge-aware grid and sign-change bracketing.

    Special cases:
    - At boundaries x=±τ, the (μ, c) mapping is handled analytically.
    - For δ=1, μ drops out in the fixed point (interpolate on c only).
    """
    nu = nu0 + k
    if nu <= 0:
        return []

    twoalpha = 2*alpha0 + k
    if twoalpha <= 0:
        return []

    Lambda_sq = max(1e-10, (1 - (1/nu)**2) / twoalpha)
    Lambda    = np.sqrt(Lambda_sq)
    tau       = (1 - 1/nu)/Lambda if Lambda > 1e-10 else 0.0
    mu_star   = 1/(nu*Lambda)

    num_points     = 1000
    edge_density   = 0.2
    center_density = 1 - 2*edge_density
    epts = int(num_points * edge_density)
    cpts = int(num_points * center_density)

    left  = np.linspace(-tau, -0.8*tau, epts, endpoint=True)
    mid   = np.linspace(-0.8*tau, 0.8*tau, cpts, endpoint=True)
    right = np.linspace(0.8*tau, tau, epts, endpoint=True)
    x_grid = np.concatenate([left[:-1], mid[:-1], right])

    def f(x):
        try:
            if abs(delta - 1.0) < 1e-10:
                a = (nu0 + k - 1)/(nu0 + k)
                b = Lambda
                c_sq = max(1e-10, (2*alpha0 + k - 1)*sigma_prev**2)
                denom = x*x*b*b - a*a
                if abs(denom) < 1e-10:
                    return np.nan
                X_minus_mu_sq = -x*x*b*b*c_sq / denom
                if X_minus_mu_sq < 0:
                    return np.nan
                X_minus_mu = np.sqrt(X_minus_mu_sq) if x >= 0 else -np.sqrt(X_minus_mu_sq)
                X = mu_prev + X_minus_mu
                sigma_sq = Lambda_sq * ((2*alpha0 + k - 1)*sigma_prev**2 + (X - mu_prev)**2)
                if sigma_sq < 1e-10:
                    return np.nan
                sigma = np.sqrt(sigma_sq)
                c = C/sigma
                S_val = get_S_value(S_df, k, None, c)
                return S_val - c - x

            # delta < 1
            if abs(x - tau) < 1e-10:
                mu = mu_star; c = 0.0
            elif abs(x + tau) < 1e-10:
                mu = -mu_star; c = 0.0
            else:
                a = (nu0 + k - 1)/(nu0 + k)
                b = Lambda
                c_sq = max(1e-10, (2*alpha0 + k - 1)*sigma_prev**2)
                denom = x*x*b*b - a*a
                if abs(denom) < 1e-10:
                    return np.nan
                X_minus_mu_sq = -x*x*b*b*c_sq / denom
                if X_minus_mu_sq < 0:
                    return np.nan
                X_minus_mu = np.sqrt(X_minus_mu_sq) if x >= 0 else -np.sqrt(X_minus_mu_sq)
                X = mu_prev + X_minus_mu
                sigma_sq = Lambda_sq * ((2*alpha0 + k - 1)*sigma_prev**2 + (X - mu_prev)**2)
                if sigma_sq < 1e-10:
                    return np.nan
                sigma = np.sqrt(sigma_sq)
                mu = (mu_prev + (X - mu_prev)/nu)/sigma
                c  = C/sigma

            S_val = get_S_value(S_df, k, mu, c)
            if np.isnan(S_val):
                return np.nan
            return S_val - c - x - (1 - delta)*mu

        except Exception:
            return np.nan

    fvals = np.array([f(x) for x in x_grid])
    mask  = ~np.isnan(fvals)
    fvals = fvals[mask]
    xg    = x_grid[mask]
    if len(fvals) < 2:
        return []

    idx = np.where(np.diff(np.sign(fvals)))[0]
    roots = []
    for i in idx:
        xl, xr = xg[i], xg[i+1]
        fl, fr = fvals[i], fvals[i+1]
        if fl * fr < 0:
            roots.append(xl - fl * (xr - xl) / (fr - fl))
    return roots


def k_doublestar(mu, sigma, alpha0, nu0, delta, beta, G, max_k=100):
    """
    Find k** (smallest k) such that S0_infinite(k, μ*) ≤ τ_k (fully responsive policy becomes optimal).
    Only valid for δ < 1.
    """
    if delta >= 1.0:
        raise ValueError("k_doublestar is defined for delta < 1.")

    S0_inf_df = S_c0_infinite(mu, sigma, alpha0, nu0, delta, beta, G)

    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1

    k = k_min
    while k <= max_k:
        if abs(nu0 + k - 1.0) < 1e-10:
            k += 1
            continue
        nu = nu0 + k
        twoalpha = 2 * alpha0 + k
        Lambda_sq = max(1e-10, (1 - (1/nu)**2) / twoalpha)
        Lambda = np.sqrt(Lambda_sq)
        tau = (1 - delta*beta/nu)/Lambda
        mu_star = 1/(nu*Lambda)

        S0 = float(get_S0_value(S0_inf_df, k, mu_star))
        if S0 < tau:
            return k
        k += 1
    return max_k + 1


def delta_star(mu, sigma, alpha0, nu0, beta, G, tol=1e-5, max_iter=100):
    """
    Critical discount δ* where S0_infinite(k_min, μ*) ≤ τ(k_min). Binary search over δ ∈ (0.8, 0.99999).
    Returns a conservative lower bound for δ*.
    """
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1
    if nu0 == 0 and alpha0 >= 0.5:
        k_min = 2

    lo, hi = 0.8, 0.99999
    it = 0
    while hi - lo > tol and it < max_iter:
        it += 1
        mid = 0.5*(lo+hi)
        nu = nu0 + k_min
        twoalpha = 2*alpha0 + k_min
        Lambda = np.sqrt((1 - (1/nu)**2) / twoalpha)
        tau = (1 - mid * beta / nu) / Lambda
        mu_star = 1/(nu*Lambda)

        S0_inf_df = S_c0_infinite(mu, sigma, alpha0, nu0, mid, beta, G)
        S0 = float(get_S0_value(S0_inf_df, k_min, mu_star))
        if S0 <= tau:
            lo = mid
        else:
            hi = mid
    return round(lo, 4)


def delta_k_star(k, mu, sigma, alpha0, nu0, beta, G, tol=5e-5, max_iter=100):
    """
    Critical δ*_k for a given k: S0_infinite(k, μ*) ≤ τ_k. Returns a conservative lower bound.
    """
    if abs(nu0 + k - 1.0) < 1e-10:
        return None

    lo = 0.99 if k > 5 else (0.95 if k > 3 else 0.88)
    hi = 0.99999
    it = 0
    while hi - lo > tol and it < max_iter:
        it += 1
        mid = 0.5*(lo+hi)
        nu = nu0 + k
        twoalpha = 2*alpha0 + k
        Lambda = np.sqrt((1 - (1/nu)**2) / twoalpha)
        tau = (1 - mid * beta / nu) / Lambda
        mu_star = 1/(nu*Lambda)

        S0_inf_df = S_c0_infinite(mu, sigma, alpha0, nu0, mid, beta, G)
        S0 = float(get_S0_value(S0_inf_df, k, mu_star))
        if S0 <= tau:
            lo = mid
        else:
            hi = mid
    return round(lo, 5)
