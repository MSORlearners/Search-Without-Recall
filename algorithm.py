"""
Algorithmic toolkit for "Search Without Recall and Gaussian Learning"
====================================================================

Purpose
-------
Provide a clean, double-blind implementation of the *algorithms* in the appendix
of the working paper “Search Without Recall and Gaussian Learning:
Structural Properties and Optimal Policies.”

The implementation follows the algorithms in the paper and uses full-support
quadrature for the predictive innovations.

Organization
------------
- Prelude: imports, numba detection
- Core helpers (pdfs, grids)
- Algorithm 1: Scaled foresight recursion S_{n,k}(·)
  - _S_fast_compute_delta1, _S_fast_compute_delta_less1, _S_fast_compute, S_fast
  - Support plane at c=0: _S_c0_compute, S_c0
  - Value extraction helpers: get_S0_value, get_S_value
- Algorithm 2: k*(n, δ) / minimum n tables
  - data_n_min
- Algorithm 3: Thresholds ξ_k (accept/continue)
  - threshold
- Algorithm 4: Infinite horizon limits and critical δ
  - S_c0_infinite, S_infinite, k_doublestar, delta_star, delta_k_star
- __main__ smoke test

Double-blind note: no author names appear in this file.
"""

# --------------------------
# Prelude: imports & numba
# --------------------------
import math
import numpy as np
import pandas as pd
from scipy.stats import t, norm
from scipy.optimize import bisect, minimize_scalar
import pandas as pd
from functools import lru_cache
from multiprocessing import Pool
import itertools
import numba
try:
    from numba import njit, prange
    _has_numba = True
except ImportError:
    _has_numba = False


# Resolution used for every numerical result reported in the paper.
PAPER_GRID_SIZE = 400


# --------------------------------------
# Core helpers: PDFs and grid generator
# --------------------------------------

@njit
def _t_pdf(x, df):
    """
    Student-t PDF (Numba-compatible) with log-space stabilization.

    Parameters
    ----------
    x : float
        Evaluation point.
    df : float
        Degrees of freedom (>0).

    Returns
    -------
    float
        f_{t_df}(x)
    """
    # Handle edge cases
    if df <= 0:
        return 0.0
        
    # Use log-space calculations for better numerical stability
    # log[Γ((df+1)/2)] - log[√(df·π)·Γ(df/2)]
    log_term1 = math.lgamma((df + 1) / 2) - (0.5 * math.log(df * math.pi) + math.lgamma(df / 2))
    
    # log[(1 + x²/df)^(-(df+1)/2)]
    log_term2 = -((df + 1) / 2) * math.log1p(x * x / df)
    
    # Combine terms and exponentiate
    return math.exp(log_term1 + log_term2)

@njit
def _norm_pdf(x):
    """Numba-compatible Normal(0,1) PDF."""
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

@njit
def _make_grids(G, rho=0.85, Z=30, ita=0.75):
    """
    Grids and full-support quadrature used by the recursions:
    - c[j] : cost grid, decreasing in j
    - z[j] : Student/Normal integration nodes on the real line
    - mu_grid[j] : bounded non-uniform μ-grid
    - dz[j] : transformed Gauss--Legendre quadrature weights in du

    The state grid remains bounded by ``Z``.  The integration grid must not be
    truncated at the same bound: under Jeffreys' prior the initial predictive
    distribution is t_2 and omitted tail contributions decay only as O(1/Z).
    We therefore write u = tan(theta) and apply Gauss--Legendre quadrature on
    theta in (-pi/2, pi/2), which covers the full real line for every G.
    """
    # Precompute the cost and state grids.
    c       = np.empty(G+2, dtype=np.float64)
    z       = np.empty(G+2, dtype=np.float64)
    mu_grid = np.empty(G+2, dtype=np.float64)
    for j in range(G+2):
        c[j]       = G * rho**j
        x = (2*j-G-1)/(G-1)
        mu_grid[j] = Z * ((1-ita)*x + ita*x**3)
    c[G] = 0.0

    # Gauss--Legendre nodes and weights on [-1, 1].  Computing them here keeps
    # the grid builder compatible with Numba and avoids a second grid-size
    # parameter in the public API.
    dz = np.empty_like(z)
    nodes = np.empty(G, dtype=np.float64)
    weights = np.empty(G, dtype=np.float64)
    half = (G + 1)//2
    for i in range(half):
        root = math.cos(math.pi*(i + 0.75)/(G + 0.5))
        derivative = 0.0
        for _ in range(100):
            p_prev = 1.0
            p_curr = root
            if G == 1:
                p_curr = root
                p_prev = 1.0
            else:
                for degree in range(2, G + 1):
                    p_next = ((2*degree - 1)*root*p_curr - (degree - 1)*p_prev)/degree
                    p_prev = p_curr
                    p_curr = p_next
            derivative = G*(root*p_curr - p_prev)/(root*root - 1.0)
            updated = root - p_curr/derivative
            if abs(updated - root) < 1e-15:
                root = updated
                break
            root = updated
        # Re-evaluate the derivative at the converged root.
        p_prev = 1.0
        p_curr = root
        if G > 1:
            for degree in range(2, G + 1):
                p_next = ((2*degree - 1)*root*p_curr - (degree - 1)*p_prev)/degree
                p_prev = p_curr
                p_curr = p_next
        derivative = G*(root*p_curr - p_prev)/(root*root - 1.0)
        weight = 2.0/((1.0 - root*root)*derivative*derivative)
        nodes[i] = -root
        nodes[G - 1 - i] = root
        weights[i] = weight
        weights[G - 1 - i] = weight

    for j in range(1, G + 1):
        theta = 0.5*math.pi*nodes[j - 1]
        z[j] = math.tan(theta)
        dz[j] = 0.5*math.pi*weights[j - 1]*(1.0 + z[j]*z[j])
    # The integration loops use only indices 1,...,G.
    z[0] = z[1]
    z[G+1] = z[G]
    dz[0] = 0.0
    dz[G+1] = 0.0
    return c, z, mu_grid, dz


# ==========================================================
# Algorithm 1 — Scaled foresight recursion S_{n,k}(·)
# ==========================================================
# Implements the backward dynamic program on (μ, c) (when δ<1)
# or on c alone (when δ=1). These correspond to the “core”
# S-recursions in the appendix pseudocode (Algorithm 1).

@njit(parallel=True)
def _S_fast_compute_delta1(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Algorithm 1 (δ = 1 case):
    Numba-compatible computation of S values (2D array over (k, c)).
    Returns raw arrays for speed; formatted later by S_fast().
    """
    # 1) build grids once
    c, z, mu_grid, dz = _make_grids(G)

    # 2) figure out k_min
    if sigma_flag == 0:
        k_min = max(math.floor(2 - 2*alpha0), 1)
    else:
        k_min = 1

    # 3) allocate S array - 2D for delta=1
    n_int = int(n)
    G_int = int(G)
    S_arr = np.zeros((n_int, G_int+2), dtype=np.float64)

    # 4) main backward recursion for delta=1
    for k in range(n_int-1, k_min-1, -1):
        df = 2*alpha0 + k
        for j_c in numba.prange(1, G_int+1):
            if k < n_int-1:
                acc = 0.0
                for j_u in numba.prange(1, G_int+1):
                    # μ_u and σ_u
                    mu_u = 0.0 if mu_flag==1 else 1.0/(nu0+k+1)
                    if sigma_flag == 0:
                        L = math.sqrt(max(0.0,(1-mu_u*mu_u)/(df+1)))
                        s = L*math.sqrt(df + z[j_u]*z[j_u])
                        pdf_val = _t_pdf(z[j_u], df)
                    else:
                        s = math.sqrt(1-mu_u*mu_u)
                        pdf_val = _norm_pdf(z[j_u])

                    # fast interpolation index via binary‐search on c
                    x = c[j_c]/s
                    # c is in descending order on j, so invert
                    i_c = G_int
                    while i_c > 1 and x > c[i_c]:
                        i_c -= 1
                    if i_c == G_int:
                        i_c = G_int-1
                    frac = (x - c[i_c])/(c[i_c+1]-c[i_c])
                    # Use 2D indexing for delta=1
                    S_u = (1-frac)*S_arr[k+1,i_c] + frac*S_arr[k+1,i_c+1]

                    val = max(0.0, (-1+delta*mu_u)*z[j_u] + s*S_u - c[j_c])
                    acc += val * pdf_val * dz[j_u]
                S_arr[k,j_c] = delta * acc
    return S_arr, k_min, c, mu_grid

@njit(parallel=True)
def _S_fast_compute_delta_less1(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Algorithm 1 (0 < δ < 1 case):
    Numba-compatible computation of S values (3D array over (k, μ, c)).
    Returns raw arrays for speed; formatted later by S_fast().
    """
    # 1) build grids once
    c, z, mu_grid, dz = _make_grids(G)

    # 2) figure out k_min
    if sigma_flag == 0:
        k_min = max(math.floor(2 - 2*alpha0), 1)
    else:
        k_min = 1

    # 3) allocate S array - 3D for delta<1
    n_int = int(n)
    G_int = int(G)
    S_arr = np.zeros((n_int, G_int+2, G_int+2), dtype=np.float64)

    # 4) main backward recursion for delta<1
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
                        new_c = c[j_c]/s

                        # find mu‐index and c‐index by binary search
                        i_mu = G_int
                        while i_mu > 1 and new_mu< mu_grid[i_mu]:
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

                        # bilinear interpolation - use 3D indexing
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

def _S_fast_compute(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Algorithm 1 (dispatcher):
    Calls the appropriate compiled kernel depending on δ.
    """
    if delta == 1.0:
        return _S_fast_compute_delta1(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)
    else:  # delta < 1
        return _S_fast_compute_delta_less1(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)

def S_fast(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Algorithm 1 (public API):
    Optimized DP recursion for S[k,j_mu,j_c] or S[k,j_c] when delta==1.
    Returns a pandas DataFrame in the same format as S().

    Notes
    -----
    - δ = 1  →  DataFrame rows indexed by k; columns are c-grid ("c=...")
    - δ < 1  →  DataFrame rows stacked by (k, c); columns are μ-grid ("mu=...")
    """
    # Compute the numerical results
    S_arr, k_min, c, mu_grid = _S_fast_compute(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)

    # Convert to DataFrame with proper labels
    if delta == 1.0:
        # For delta=1, we have a 2D array (k, c)
        df = pd.DataFrame(S_arr[k_min:n, 1:G+1])
        # Set column labels
        column_labels = [f'c={float(c[j])!r}' for j in range(1, G+1)]
        df.columns = column_labels
        # Set row labels
        row_labels = ['n=%d, k_min=%d, k=%d' % (n, k_min, k) for k in range(k_min, n)]
        df.index = row_labels
    else:
        # For delta<1, we have a 3D array and need to reshape it into a 2D DataFrame
        # Each row will be a combination of k and c, columns will be mu values
        rows = []
        row_labels = []
        for k in range(k_min, n):
            for j_c in range(1, G+1):
                rows.append(S_arr[k, 1:G+1, j_c])  # Only use indices 1 to G+1
                row_labels.append(f'k={k}, c={float(c[j_c])!r}')
        df = pd.DataFrame(rows)
        # Set column labels
        column_labels = [f'mu={float(mu_grid[j_mu])!r}' for j_mu in range(1, G+1)]
        df.columns = column_labels
        # Set row labels
        df.index = row_labels

    return df, c, mu_grid


# ------------------------------------------------------------
# Support plane at c = 0 (used in Algos 2 & 4, and by helpers)
# ------------------------------------------------------------

@njit(parallel=True)
def _S_c0_compute(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Compute S(k, μ, c=0) on the μ-grid (δ<1) or the c≈0 slice (δ=1).
    Returns a raw array; formatting happens in S_c0().
    """
    # 1) build grids once (no need for c grid)
    _, z, mu_grid, dz = _make_grids(G)
    # 2) figure out k_min
    if sigma_flag == 0:
        k_min = max(math.floor(2 - 2*alpha0), 1)
    else:
        k_min = 1

    # 3) allocate S array - only need 2D shape (k, mu)
    # Ensure n and G are integers for Numba compatibility
    n_int = int(n)
    G_int = int(G)
    S_arr = np.zeros((n_int, G_int+2), dtype=np.float64)

    # 4) main backward recursion
    for k in range(n_int-1, k_min-1, -1):
        df = 2*alpha0 + k
        
        # Handle delta=1 case separately
        if delta == 1.0:
            # For delta=1, we still follow the algorithm structure but with c=0
            # We need to compute for j_c (even though c=0) to maintain algorithm structure
            for j_c in range(1, 2):  # Only one iteration since c=0, but keep the structure
                if k < n_int-1:
                    acc = 0.0
                    for j_u in numba.prange(1, G_int+1):
                        if mu_flag == 0:
                            mu_u = 1/(nu0+k+1)
                        elif mu_flag == 1:
                            mu_u = 0
                        
                        if sigma_flag == 0:
                            # σ_u = Λ_{k+1} * sqrt(2α_0 + k + z[j_u]^2)
                            Lambda_k1 = math.sqrt(max(0.0, (1-(1/(nu0+k+1))**2)/(2*alpha0+k+1)))
                            s = Lambda_k1 * math.sqrt(2*alpha0 + k + z[j_u]*z[j_u])
                            pdf_val = _t_pdf(z[j_u], 2*alpha0+k)
                        elif sigma_flag == 1:
                            s = math.sqrt(1-(1/(nu0+k+1))**2)
                            pdf_val = _norm_pdf(z[j_u])
                        
                        # For c=0: ĉ = 0/σ_u = 0, so S^+ = S[k+1, 0] = S[k+1, 1] (index 1 corresponds to c≈0)
                        # g = max{0, -(ν_0+k)/(ν_0+k+1) * z[j_u] + σ_u * S^+ - c[j_c]}
                        # Since c[j_c] = 0 for c=0 case:
                        integrand = max(
                            0.0,
                            -(nu0 + k)/(nu0 + k + 1) * z[j_u] + s * S_arr[k+1, 1]
                        )
                        # S[k,j_c] += g * f_{2α}(z[j_u]) * (z[j_u+1] - z[j_u-1])/2
                        acc += integrand * pdf_val * dz[j_u]
                    
                    # Store result at index 1 (corresponding to c≈0)
                    S_arr[k, 1] = acc
        else:
            # For delta<1, we need to compute for each mu value
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

                        # find mu‐index by binary search
                        i_mu = G_int
                        while i_mu > 1 and new_mu < mu_grid[i_mu]:
                            i_mu -= 1
                        if i_mu == G_int:
                            i_mu = G_int-1
                        frac_mu = (new_mu - mu_grid[i_mu])/(mu_grid[i_mu+1]-mu_grid[i_mu])

                        # linear interpolation in mu only
                        S_interp = (1-frac_mu)*S_arr[k+1, i_mu] + frac_mu*S_arr[k+1, i_mu+1]

                        integrand = max(
                            0.0,
                            -(1 - delta/(nu0 + k + 1)) * z[j_u]
                            - (1 - delta)*mu0
                            + s*S_interp
                        )
                        acc += integrand * pdf_val * dz[j_u]
                    S_arr[k, j_mu] = delta * acc
                # else boundary zero
    return S_arr, k_min, mu_grid

def S_c0(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G):
    """
    Support routine: S(k, μ, c=0), formatted as a DataFrame.

    - δ = 1 → DataFrame with single column 'S0' indexed by k
    - δ < 1 → DataFrame with μ-grid columns ('mu=...') indexed by k
    """
    # Compute the numerical results
    S_arr, k_min, mu_grid = _S_c0_compute(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)
    
    # Convert to DataFrame with proper labels based on delta value
    if delta == 1:
        # For delta=1, we have a 2D array but only use index 1 (corresponding to c≈0)
        df = pd.DataFrame(S_arr[k_min:n, 1:2])  # Extract column 1 as a 2D slice
        df.columns = ['S0']
        
        # Set row labels as k values only
        row_labels = [f'k={k}' for k in range(k_min, n)]
        df.index = row_labels
    else:
        # For delta<1, we need the full 2D DataFrame
        df = pd.DataFrame(S_arr[k_min:n, 1:G+1])
        
        # Create column labels with proper numerical ordering
        # First create a list of (mu_value, label) pairs
        mu_labels = [(mu_grid[j_mu], f'mu={float(mu_grid[j_mu])!r}') for j_mu in range(1, G+1)]
        
        # Extract just the labels in sorted order
        column_labels = [label for _, label in mu_labels]
        # Set column labels as mu values
        df.columns = column_labels
        
        # Set row labels as k values
        row_labels = [f'k={k}' for k in range(k_min, n)]
        df.index = row_labels
    # Truncate mu_grid to only include the values used in the DataFrame
    # This makes the returned mu_grid consistent with the DataFrame columns
    if delta < 1:
        mu_grid = mu_grid[1:G+1]
    else:
        # For delta=1, we don't need the full mu_grid
        mu_grid = np.array([0.0])  # Just a placeholder
    return df, mu_grid


# ---------------------------------------------
# Helpers: value extraction & interpolation
# ---------------------------------------------

def _grid_bracket(grid, value):
    """Return positions and a weight, using the end interval for extrapolation."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("Grid coordinates must be a nonempty finite vector.")
    if value is None or not np.isfinite(value):
        raise ValueError("A finite interpolation coordinate is required.")
    order = np.argsort(grid)
    coordinates = grid[order]
    if np.any(np.diff(coordinates) <= 0):
        raise ValueError("Duplicate grid coordinates; recompute tables with unrounded labels.")
    if len(grid) == 1:
        if value != coordinates[0]:
            raise ValueError("Cannot extrapolate from a single grid point.")
        return int(order[0]), int(order[0]), 0.0
    i = int(np.clip(np.searchsorted(coordinates, value) - 1, 0, len(grid) - 2))
    weight = (value - coordinates[i]) / (coordinates[i + 1] - coordinates[i])
    return int(order[i]), int(order[i + 1]), float(weight)


def _label_fields(label):
    return dict(part.strip().split("=", 1) for part in label.split(","))


def _check_table_labels(table):
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError("Duplicate table labels; recompute tables with unrounded labels.")


def get_S0_value(S0_df_tuple, k, mu_val):
    """Read S(k, mu, 0), with linear interpolation and end-interval extrapolation.

    Accepts S_c0's (DataFrame, exact mean grid) or a DataFrame alone.
    Missing stages and ambiguous legacy tables raise an error.
    """
    if isinstance(S0_df_tuple, tuple):
        S0_df, mu_grid = S0_df_tuple
    else:
        S0_df, mu_grid = S0_df_tuple, None
    _check_table_labels(S0_df)
    row = S0_df.loc[f'k={k}']
    if list(S0_df.columns) == ['S0']:
        return float(row.iloc[0])
    if mu_grid is None:
        mu_grid = [float(_label_fields(col)['mu']) for col in S0_df.columns]
    if len(mu_grid) != len(row):
        raise ValueError("Mean grid length does not match the table columns.")
    lower, upper, weight = _grid_bracket(mu_grid, mu_val)
    return float((1 - weight) * row.iloc[lower] + weight * row.iloc[upper])


def get_S_value(S_df, k, mu_val, c_val):
    """Read S(k, mu, c) by (bi)linear interpolation.

    Labels retain full-precision coordinates. Out-of-grid values use the
    adjacent end interval, as in the Bellman kernels. Rounded legacy tables
    with duplicate labels must be recomputed.
    """
    _check_table_labels(S_df)
    rows = [i for i, label in enumerate(S_df.index)
            if int(_label_fields(label)['k']) == k]
    if not rows:
        raise KeyError(f"Stage k={k} is not in the table.")
    if S_df.columns[0].startswith('mu='):
        means = [float(_label_fields(col)['mu']) for col in S_df.columns]
        costs = [float(_label_fields(S_df.index[i])['c']) for i in rows]
        ml, mh, wm = _grid_bracket(means, mu_val)
        cl, ch, wc = _grid_bracket(costs, c_val)
        lower = (1 - wm) * S_df.iloc[rows[cl], ml] + wm * S_df.iloc[rows[cl], mh]
        upper = (1 - wm) * S_df.iloc[rows[ch], ml] + wm * S_df.iloc[rows[ch], mh]
        return float((1 - wc) * lower + wc * upper)
    if len(rows) != 1:
        raise ValueError(f"Multiple rows found for stage k={k}.")
    costs = [float(_label_fields(col)['c']) for col in S_df.columns]
    lower, upper, weight = _grid_bracket(costs, c_val)
    return float((1 - weight) * S_df.iloc[rows[0], lower]
                 + weight * S_df.iloc[rows[0], upper])


# =========================================
# Algorithm 2 — k*(n, δ) / maximum-horizon table
# =========================================
# This routine scans horizons n for each k and reports the last horizon
# associated with that critical sample size, matching Algorithm 2.

def data_n_min(mu, sigma, alpha0, nu0, delta, beta, G, k_upper, *,
               n_upper=None, convergence_tol=1e-6):
    """Scan for the last horizon associated with each k (Algorithm 2).

    A finite n_max is reported only after observing a strict crossing.
    Otherwise n_max is NaN; status distinguishes a horizon cap, numerical
    stabilization, and an excluded stage. n_checked records the last horizon
    tested. Stabilization is not a certified infinite-horizon bound.
    Set convergence_tol=0 to disable the stabilization shortcut.
    """
    if not 0 < delta <= 1:
        raise ValueError("delta must lie in (0, 1].")
    if convergence_tol < 0:
        raise ValueError("convergence_tol must be nonnegative.")
    k_min = max(math.floor(2 - 2 * alpha0), 1) if sigma == 0 else 1
    if n_upper is None:
        n_upper = 500 if delta <= 0.998 else 3000
    results = []
    for k in range(k_min, k_upper + 1):
        if nu0 + k <= 1:
            results.append([k, np.nan, 'excluded-stage', np.nan])
            continue
        omega = math.sqrt((2 * alpha0 + k) / ((nu0 + k) ** 2 - 1))
        tau = (nu0 + k - delta) * omega
        previous = None
        checked = np.nan
        status, n_max = 'horizon-cap', np.nan
        for n in range(k + 2, n_upper + 1):
            table = S_c0(mu, sigma, alpha0, nu0, n, delta, beta, G)
            value = get_S0_value(table, k, None if delta == 1 else omega)
            checked = n
            crossed = value > tau
            if k == 1 and not crossed:
                omega2 = math.sqrt((2 * alpha0 + 2) / ((nu0 + 2) ** 2 - 1))
                crossed = get_S0_value(table, 2, omega2) > (nu0 + 2 - delta) * omega2
            # Crossing must be checked before a small successive-horizon change.
            if crossed:
                n_max, status = n - 1, 'crossing-found'
                break
            if (delta < 1 and previous is not None
                    and abs(value - previous) < convergence_tol):
                status = 'numerical-stabilization'
                break
            previous = value
        results.append([k, n_max, status, checked])
    df = pd.DataFrame(results, columns=['k', 'n_max', 'status', 'n_checked'])
    df.index.name = 'Index'
    return df


# ============================================
# Algorithm 3 — Thresholds ξ_k (accept/continue)
# ============================================

def threshold(k, mu_prev, sigma_prev, alpha0, nu0, delta, S_df, C,
              tol=1e-8, grid_size=1001):
    """Compute the offer values at which stopping and continuation are equal.

    The gain from continuation is evaluated as a function of the current offer
    ``X``. Sign-changing roots are refined by bisection; roots that fall on the
    grid are retained directly. If neither is found, convex minimization checks
    for a tangent root or a negative minimum between grid points. A negative
    minimum brackets a root with each positive grid endpoint. The grid range
    must contain the crossings. The returned list contains zero, one, or two
    offer-value thresholds in increasing order.
    """
    nu = nu0 + k
    twoalpha = 2*alpha0 + k
    if nu <= 1 or twoalpha <= 0 or sigma_prev < 0 or C < 0:
        return []
    if not 0 < delta <= 1 or tol <= 0 or grid_size < 3:
        raise ValueError("Require 0 < delta <= 1, tol > 0, and grid_size >= 3")

    Lambda_squared = (1 - (1/nu)**2) / twoalpha
    if Lambda_squared <= 0:
        return []
    Lambda = math.sqrt(Lambda_squared)
    base_scale_squared = max(1e-12, (twoalpha - 1)*sigma_prev**2)
    base_scale = math.sqrt(base_scale_squared)
    is_delta_one = delta == 1.0

    def gain(X):
        innovation = X - mu_prev
        mu_post = mu_prev + innovation/nu
        sigma_post = Lambda*math.sqrt(base_scale_squared + innovation**2)
        scaled_cost = C/sigma_post
        scaled_mu = mu_post/sigma_post
        S_value = get_S_value(
            S_df, k, None if is_delta_one else scaled_mu, scaled_cost
        )
        if S_value is None or not np.isfinite(S_value):
            raise ValueError("Threshold search requires finite foresight values.")
        return delta*mu_post - X + sigma_post*S_value - C

    # X = mu_{k-1} + base_scale*tan(theta) maps the finite theta interval
    # onto the full offer line. The small endpoint margin represents the two
    # asymptotic tails without evaluating at infinity.
    endpoint_margin = 1e-6
    theta = np.linspace(
        -0.5*math.pi + endpoint_margin,
        0.5*math.pi - endpoint_margin,
        int(grid_size),
    )
    offer_grid = mu_prev + base_scale*np.tan(theta)
    gain_values = np.array([gain(X) for X in offer_grid])
    valid = np.isfinite(gain_values)
    offer_grid = offer_grid[valid]
    gain_values = gain_values[valid]
    if len(gain_values) < 3:
        return []

    roots = [
        float(X) for X, value in zip(offer_grid, gain_values)
        if abs(value) <= tol
    ]
    for i in range(len(gain_values) - 1):
        if gain_values[i]*gain_values[i + 1] < 0:
            root = bisect(
                gain, offer_grid[i], offer_grid[i + 1],
                xtol=tol, rtol=max(tol, 4*np.finfo(float).eps),
            )
            roots.append(float(root))

    # A tangent root or two roots in one grid interval need not change signs
    # at the sampled points. Include minima next to either grid boundary.
    if not roots:
        i_min = int(np.argmin(gain_values))
        result = minimize_scalar(
            gain,
            bounds=(
                offer_grid[max(0, i_min - 1)],
                offer_grid[min(len(offer_grid) - 1, i_min + 1)],
            ),
            method="bounded",
            options={"xatol": tol},
        )
        if result.success and np.isfinite(result.fun):
            if abs(result.fun) <= tol:
                roots.append(float(result.x))
            elif result.fun < -tol:
                for endpoint, value in (
                    (offer_grid[0], gain_values[0]),
                    (offer_grid[-1], gain_values[-1]),
                ):
                    if value > 0:
                        left, right = sorted((endpoint, result.x))
                        roots.append(float(bisect(
                            gain, left, right, xtol=tol,
                            rtol=max(tol, 4*np.finfo(float).eps),
                        )))

    roots.sort()
    deduped = []
    for root in roots:
        if not deduped or abs(root - deduped[-1]) > 10*tol*max(1.0, abs(root)):
            deduped.append(root)
    return deduped[:2]


# =====================================================
# Algorithm 4 — Infinite horizon and critical δ values
# =====================================================

def S_c0_infinite(
    mu, sigma, alpha0, nu0, delta, beta, G, k_upper=None, tol=1e-5, max_iter=8
):
    """
    Algorithm 4 (c=0 slice):
    Compute the infinite-horizon S(k, μ, 0) by doubling the truncation horizon.

    Convergence is checked in sup norm over the full μ-grid and every stage
    from k_min through k_upper, as in Algorithm 4 of the paper.  The legacy
    ``beta`` argument is retained for API compatibility but is not a model
    parameter.
    """
    # Determine k_min
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1
    
    if not 0 < delta < 1:
        raise ValueError("S_c0_infinite requires 0 < delta < 1")
    if k_upper is None:
        k_upper = k_min

    # Start with a horizon scaled to the effective discount horizon.
    if 0.999<=delta<1:
        n_initial = min(int(1/(1-delta)), 10000)  # Cap at 10000 to avoid memory issues
    elif 0.995<=delta<0.999:
        n_initial = 2000
    elif 0.990<=delta<0.995:
        n_initial = 1000
    else:
        n_initial = 500
    n = max(int(n_initial), int(k_upper) + 2)
    S_old, mu_grid = S_c0(mu, sigma, alpha0, nu0, n, delta, beta, G)
    rows = [f'k={k}' for k in range(k_min, int(k_upper) + 1)]

    for iteration in range(1, max_iter + 1):
        new_n = 2*n
        S_new, mu_grid = S_c0(mu, sigma, alpha0, nu0, new_n, delta, beta, G)
        diff = float(np.max(np.abs(
            S_new.loc[rows].to_numpy() - S_old.loc[rows].to_numpy()
        )))
        print(f"iteration={iteration}, n={new_n}, sup_diff={diff:.6g}")
        if diff <= tol:
            return S_new, mu_grid
        n = new_n
        S_old = S_new

    raise RuntimeError(
        f"Infinite-horizon recursion did not converge within {max_iter} doublings; "
        f"last horizon={n}, sup-norm difference={diff:.6g}"
    )

def S_infinite(
    mu, sigma, alpha0, nu0, delta, beta, G, k_upper, tol=1e-5, max_iter=8
):
    """
    Algorithm 4 (general S):
    Compute the infinite-horizon S(k, μ, c) by doubling the truncation horizon
    until all requested stages stabilize in sup norm.
    """
    # Determine k_min
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1
    
    if not 0 < delta < 1:
        raise ValueError("S_infinite requires 0 < delta < 1")

    n = max(500, int(k_upper) + 2)
    S_old, _, mu_grid = S_fast(mu, sigma, alpha0, nu0, n, delta, beta, G)
    row_mask = np.array([
        any(label.startswith(f'k={k},') for k in range(k_min, int(k_upper) + 1))
        for label in S_old.index
    ])
    rows = S_old.index[row_mask]

    for iteration in range(1, max_iter + 1):
        new_n = 2*n
        S_new, _, mu_grid = S_fast(
            mu, sigma, alpha0, nu0, new_n, delta, beta, G
        )
        diff = float(np.max(np.abs(
            S_new.loc[rows].to_numpy() - S_old.loc[rows].to_numpy()
        )))
        print(f"iteration={iteration}, n={new_n}, sup_diff={diff:.6g}")
        if diff <= tol:
            return S_new, mu_grid
        n = new_n
        S_old = S_new

    raise RuntimeError(
        f"Infinite-horizon recursion did not converge within {max_iter} doublings; "
        f"last horizon={n}, sup-norm difference={diff:.6g}"
    )

def k_doublestar(mu, sigma, alpha0, nu0, delta, beta, G, max_k=100):
    """Return the first admissible k with S_infinite(k, omega_k, 0) <= tau_k.

    Stage 1 is omitted when stage 2 probes, as in Algorithm 4.
    Returns None when no crossing is found through max_k; convergence or
    lookup failures propagate rather than being reported as a boundary.
    """
    if not 0 < delta < 1:
        raise ValueError("This function requires 0 < delta < 1.")
    k_min = max(math.floor(2 - 2 * alpha0), 1) if sigma == 0 else 1
    if max_k < k_min:
        return None
    table = S_c0_infinite(
        mu, sigma, alpha0, nu0, delta, beta, G, k_upper=max(max_k, 2)
    )

    def gap(k):
        omega = math.sqrt((2 * alpha0 + k) / ((nu0 + k) ** 2 - 1))
        return get_S0_value(table, k, omega) - (nu0 + k - delta) * omega

    for k in range(k_min, max_k + 1):
        if nu0 + k <= 1:
            continue
        if k == 1 and gap(2) > 0:
            continue
        if gap(k) <= 0:
            return k
    return None

def delta_star(mu, sigma, alpha0, nu0, beta, G, tol=1e-5, max_iter=100):
    """
    Algorithm 4 (critical discount factor, global):
    Find δ* as the largest δ such that S0_infinite(k_min) ≤ τ_{k_min}.
    Binary search with S_c0_infinite.
    """
    print(f"Computing delta* for alpha0={alpha0}, nu0={nu0}...")
    
    # Determine k_min based on sigma
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1
        
    # Special case: exclude fully responsive case for nu0=0 and alpha0>=0.5
    if nu0 == 0 and alpha0 >= 0.5:
        k_min = 2
    
    # Binary search bounds
    delta_low = 1e-6
    delta_high = 1 - 1e-6
    
    # Binary search
    iterations = 0
    
    while delta_high - delta_low > tol and iterations < max_iter:
        iterations += 1
        delta_mid = (delta_low + delta_high) / 2
        print(f"Iteration {iterations}: Testing delta = {delta_mid:.6f}")
        
        try:
            # Calculate tau at current delta
            nu = nu0 + k_min
            twoalpha = 2 * alpha0 + k_min
            Lambda = np.sqrt((1 - (1/nu)**2) / twoalpha)
            tau = (1 - delta_mid / nu) / Lambda
            mu_star = 1/(nu*Lambda)
            
            # Get S0_infinite for current delta
            S0_inf_df = S_c0_infinite(
                mu, sigma, alpha0, nu0, delta_mid, beta, G,
                k_upper=k_min, tol=tol
            )
            S0_val = get_S0_value(S0_inf_df, k_min, mu_star)
            S0 = float(S0_val)
            
            print(f"   S0={S0:.6f}, tau={tau:.6f}, diff={S0-tau:.6f}")
            
            # Compare S0 with tau
            if S0 <= tau:
                # We can potentially use a higher delta
                delta_low = delta_mid
            else:
                # Need to use a lower delta
                delta_high = delta_mid
                
        except Exception as e:
            raise RuntimeError(
                f"Discount search failed at delta={delta_mid}; no boundary was inferred."
            ) from e
    
    return 0.5*(delta_low + delta_high)

def delta_k_star(k, mu, sigma, alpha0, nu0, beta, G, tol=5e-5, max_iter=100):
    """
    Algorithm 4 (critical discount factor at fixed k):
    Find δ*_k via binary search s.t. S0_infinite(k) ≤ τ_k at δ=δ*_k.
    """
    print(f"Computing delta* for k={k}, alpha0={alpha0}, nu0={nu0}...")
    
    # Check for invalid k values
    if abs(nu0 + k - 1.0) < 1e-10:
        print(f"Warning: k={k} results in nu0+k=1, which is invalid")
        return None
    
    # Binary search bounds
    delta_low = 1e-6
    delta_high = 1 - 1e-6
    
    # Binary search
    iterations = 0
    S0=1
    tau=0
    while delta_high - delta_low > tol and iterations < max_iter:
        iterations += 1
        print(f"ITERATION {iterations}: delta_low={delta_low:.6f}, delta_high={delta_high:.6f}")
        delta_mid = (delta_low + delta_high) / 2
        print(f"ITERATION {iterations}: Testing delta = {delta_mid:.6f}")
        
        try:
            # Calculate tau at current delta for the given k
            nu = nu0 + k
            twoalpha = 2 * alpha0 + k
            Lambda = np.sqrt((1 - (1/nu)**2) / twoalpha)
            tau = (1 - delta_mid / nu) / Lambda
            mu_star = 1/(nu*Lambda)
            
            # Get S0_infinite for current delta
            S0_inf_df = S_c0_infinite(
                mu, sigma, alpha0, nu0, delta_mid, beta, G,
                k_upper=k, tol=tol
            )
            S0_val = get_S0_value(S0_inf_df, k, mu_star)
            S0 = float(S0_val)
            print(f"   S0={S0:.6f}, tau={tau:.6f}, diff={S0-tau:.6f}")
            
            # Compare S0 with tau
            if S0 <= tau:
                # We can potentially use a higher delta
                delta_low = delta_mid
            else:
                # Need to use a lower delta
                delta_high = delta_mid
                
        except Exception as e:
            raise RuntimeError(
                f"Discount search failed at delta={delta_mid}; no boundary was inferred."
            ) from e
    
    return 0.5*(delta_low + delta_high)



# ---------------------------------------------------
# JIT decoration check (left as in the original code)
# ---------------------------------------------------
# If you have Numba installed, you can JIT the inner loops:
if _has_numba:
    pass  # Functions are already decorated with @njit


# --------------------------
# __main__ smoke test block
# --------------------------
if __name__ == '__main__':
    import time
    # Small smoke-test parameters (not the paper's reported numerical setting).
    alpha0 = -0.5
    nu0 = 0
    mu = 0
    sigma = 0
    beta = 1
    G = 40
    k_upper =14
    delta=0.9
    n=100
    
    # test S_fast with time
    start_time = time.time()
    S_fast(mu, sigma, alpha0, nu0, n, delta, beta, G)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
