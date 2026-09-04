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
    if abs(delta - 1.0) < 1e-10:  # delta == 1
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
        column_labels = ['c=%.1f' % c[j] for j in range(1, G+1)]
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
                row_labels.append('k=%d, c=%.6f' % (k, c[j_c]))
        df = pd.DataFrame(rows)
        # Set column labels
        column_labels = ['mu=%.6f' % mu_grid[j_mu] for j_mu in range(1, G+1)]
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
        mu_labels = [(mu_grid[j_mu], f'mu={mu_grid[j_mu]:.6f}') for j_mu in range(1, G+1)]
        
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

def get_S0_value(S0_df_tuple, k, mu_val):
    """
    Helper: extract S0(k, μ) from S_c0(...) output using (at most) linear
    interpolation in μ (when δ<1). Accepts (DataFrame, μ-grid) or just DataFrame.
    """
    # Handle both DataFrame and (DataFrame, mu_grid) tuple formats
    if isinstance(S0_df_tuple, tuple) and len(S0_df_tuple) == 2:
        S0_df, mu_grid = S0_df_tuple
    else:
        S0_df = S0_df_tuple
        mu_grid = None
    
    # Check if DataFrame has a simple structure (delta=1 case) or complex structure (delta<1 case)
    is_delta_one = len(S0_df.columns) == 1 and S0_df.columns[0] == 'S0'
    
    # Get available k values
    available_k = sorted(set(int(label.split('=')[1]) for label in S0_df.index))
    if k not in available_k:
        print(f"Error: k={k} not found in available k values: {available_k}")
        return None
    
    # Handle delta=1 case
    if is_delta_one:
        # For delta=1, we directly return the value for this k
        # mu_val is ignored since it doesn't affect the result
        try:
            val = S0_df.loc[f'k={k}', 'S0']
            return val.iloc[0] if hasattr(val, 'iloc') else val
        except (KeyError, ValueError) as e:
            print(f"Error accessing k={k} in delta=1 format: {e}")
            return 0.0
    
    # Handle delta<1 case with exact mu_grid for more precise interpolation
    else:
        if mu_val is None:
            raise ValueError("mu_val cannot be None for delta<1 case")
            
        # Using mu_grid for precise interpolation if available
        if mu_grid is not None:
            # Find the appropriate mu indices for interpolation
            i_mu = len(mu_grid) - 1
            while i_mu > 1 and mu_val < mu_grid[i_mu]:
                i_mu -= 1
                
            # Ensure we don't go out of bounds
            if i_mu >= len(mu_grid) - 1:
                i_mu = len(mu_grid) - 2
                
            mu_lower = mu_grid[i_mu]
            mu_upper = mu_grid[i_mu + 1]
            
            # Calculate interpolation weight
            w_mu = (mu_val - mu_lower) / (mu_upper - mu_lower) if mu_upper != mu_lower else 0
            
            # Get the values from the DataFrame
            try:
                S_lower = S0_df.loc[f'k={k}', f'mu={mu_lower:.6f}']
                S_upper = S0_df.loc[f'k={k}', f'mu={mu_upper:.6f}']
                
                # Extract scalar values from Series if needed
                S_lower = S_lower.iloc[0] if hasattr(S_lower, 'iloc') else S_lower
                S_upper = S_upper.iloc[0] if hasattr(S_upper, 'iloc') else S_upper
                
                # Perform linear interpolation
                S_interp = S_lower * (1 - w_mu) + S_upper * w_mu
                return S_interp
            except (KeyError, ValueError) as e:
                print(f"Error in interpolation using mu_grid for k={k}, mu={mu_val}: {e}")
                # Try to find closest mu value as fallback
                try:
                    closest_mu = min(mu_grid[1:len(mu_grid)-1], key=lambda x: abs(x - mu_val))
                    val = S0_df.loc[f'k={k}', f'mu={closest_mu:.6f}']
                    return val.iloc[0] if hasattr(val, 'iloc') else val
                except (KeyError, ValueError) as e:
                    print(f"Fallback using mu_grid also failed: {e}")
                    return 0.0
        else:
            # Fallback to using column headers if mu_grid is not available
            mu_columns = [float(col.split('=')[1]) for col in S0_df.columns]
            mu_lower = max([m for m in mu_columns if m <= mu_val], default=mu_columns[0])
            mu_upper = min([m for m in mu_columns if m >= mu_val], default=mu_columns[-1])
            
            if mu_lower == mu_upper:
                mu_upper = mu_columns[min(mu_columns.index(mu_lower) + 1, len(mu_columns)-1)]
            
            # Get the values for linear interpolation
            try:
                S_lower = S0_df.loc[f'k={k}', f'mu={mu_lower:.6f}']
                S_upper = S0_df.loc[f'k={k}', f'mu={mu_upper:.6f}']
                
                # Extract scalar values from Series if needed
                S_lower = S_lower.iloc[0] if hasattr(S_lower, 'iloc') else S_lower
                S_upper = S_upper.iloc[0] if hasattr(S_upper, 'iloc') else S_upper
                
                # Calculate interpolation weight
                w_mu = (mu_val - mu_lower) / (mu_upper - mu_lower) if mu_upper != mu_lower else 0
                
                # Perform linear interpolation
                S_interp = S_lower * (1 - w_mu) + S_upper * w_mu
                return S_interp
            except (KeyError, ValueError) as e:
                print(f"Error in interpolation using column headers for k={k}, mu={mu_val}: {e}")
                # Try direct access as a fallback
                try:
                    closest_mu = min(mu_columns, key=lambda x: abs(x - mu_val))
                    val = S0_df.loc[f'k={k}', f'mu={closest_mu:.6f}']
                    return val.iloc[0] if hasattr(val, 'iloc') else val
                except (KeyError, ValueError) as e:
                    print(f"All fallbacks failed: {e}")
                    return 0.0

def get_S_value(S_df, k, mu_val, c_val):
    """
    Helper: extract S(k, μ, c) from S_fast(...) output by
    (bi)linear interpolation. Supports both δ=1 (c-only columns)
    and δ<1 (μ-columns with stacked (k,c) rows).
    """
    # Handle delta < 1 case with mu columns
    if 'mu=' in S_df.columns[0]:
        if mu_val is None:
            raise ValueError("mu_val cannot be None when delta < 1")
        
        # Get available k values
        available_k = sorted(set(int(label.split('k=')[1].split(',')[0]) 
                            for label in S_df.index if 'k=' in label))
        
        if k not in available_k:
            print(f"Error: k={k} not found in available k values: {available_k}")
            return None
            
        # Get available mu values from columns
        mu_columns = [float(col.split('=')[1]) for col in S_df.columns]
        mu_lower = max([m for m in mu_columns if m <= mu_val], default=mu_columns[0])
        mu_upper = min([m for m in mu_columns if m >= mu_val], default=mu_columns[-1])
        
        if mu_lower == mu_upper:
            mu_upper = mu_columns[min(mu_columns.index(mu_lower) + 1, len(mu_columns)-1)]
        
        # Get available c values from index for this k
        c_values = [float(label.split('c=')[1]) for label in S_df.index 
                   if f'k={k}' in label and 'c=' in label]
        c_values = sorted(set(c_values))
        
        c_lower = max([c for c in c_values if c <= c_val], default=c_values[0])
        c_upper = min([c for c in c_values if c >= c_val], default=c_values[-1])
        
        if c_lower == c_upper:
            c_upper = c_values[min(c_values.index(c_lower) + 1, len(c_values)-1)]
        
        # Safe lookup function to handle potential KeyError and Series returns
        def get_value(k_val, c_val, mu_val):
            try:
                row_label = f'k={k_val}, c={c_val:.6f}'
                col_label = f'mu={mu_val:.6f}'
                value = S_df.loc[row_label, col_label]
                return value.iloc[0] if hasattr(value, 'iloc') else value
            except (KeyError, ValueError) as e:
                print(f"Warning: Error accessing {row_label}, {col_label}: {e}")
                return 0.0
        
        # Get the four corner values for bilinear interpolation
        S_ll = get_value(k, c_lower, mu_lower)
        S_lu = get_value(k, c_lower, mu_upper)
        S_ul = get_value(k, c_upper, mu_lower)
        S_uu = get_value(k, c_upper, mu_upper)
        
        # Calculate interpolation weights
        w_mu = (mu_val - mu_lower) / (mu_upper - mu_lower) if mu_upper != mu_lower else 0
        w_c = (c_val - c_lower) / (c_upper - c_lower) if c_upper != c_lower else 0
        
        # Perform bilinear interpolation
        S_interp = (
            S_ll * (1 - w_mu) * (1 - w_c) +
            S_lu * w_mu * (1 - w_c) +
            S_ul * (1 - w_mu) * w_c +
            S_uu * w_mu * w_c
        )
        
        return float(S_interp)  # Ensure we return a scalar float
    
    # Handle delta = 1 case with c columns
    else:
        # Extract n and k_min from the first row index to handle 'n=%d, k_min=%d, k=%d' format
        first_row = S_df.index[0]
        parts = first_row.split(', ')
        n_part = parts[0]
        k_min_part = parts[1] if len(parts) > 1 else None
        
        # Find the right row based on k value
        row_matches = [i for i, idx in enumerate(S_df.index) if f', k={k}' in idx]
        if not row_matches:
            print(f"Error: No row found for k={k} in delta=1 format")
            return None
            
        row_idx = S_df.index[row_matches[0]]
        
        # Get available c values from columns
        c_columns = [float(col.split('=')[1]) for col in S_df.columns]
        c_lower = max([c for c in c_columns if c <= c_val], default=c_columns[0])
        c_upper = min([c for c in c_columns if c >= c_val], default=c_columns[-1])
        
        if c_lower == c_upper:
            c_upper = c_columns[min(c_columns.index(c_lower) + 1, len(c_columns)-1)]
        
        # Safe lookup function
        def get_value(row_idx, c_val):
            try:
                col_label = f'c={c_val:.1f}'
                value = S_df.loc[row_idx, col_label]
                return value.iloc[0] if hasattr(value, 'iloc') else value
            except (KeyError, ValueError) as e:
                print(f"Warning: Error accessing {row_idx}, {col_label}: {e}")
                return 0.0
        
        # Get the values for linear interpolation
        S_lower = get_value(row_idx, c_lower)
        S_upper = get_value(row_idx, c_upper)
        
        # Calculate interpolation weight
        w_c = (c_val - c_lower) / (c_upper - c_lower) if c_upper != c_lower else 0
        
        # Perform linear interpolation
        S_interp = S_lower * (1 - w_c) + S_upper * w_c
        
        return float(S_interp)  # Ensure we return a scalar float


# =========================================
# Algorithm 2 — k*(n, δ) / maximum-horizon table
# =========================================
# This routine scans horizons n for each k and reports the last horizon
# associated with that critical sample size, matching Algorithm 2.

def data_n_min(mu, sigma, alpha0, nu0, delta, beta, G, k_upper):    
    """
    Algorithm 2:
    Compute the maximum horizon associated with each k in the finite-horizon
    k*(n, δ) lookup table.  The function name is retained for compatibility.

    Returns
    -------
    pandas.DataFrame
        Columns ['k', 'n_max'] with strings 'Null-*' when not applicable.
    """
    # Determine k_min based on sigma
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1
    
    # For all other cases, proceed with normal calculation
    k_value = k_min
    n_start = k_value + 2
    Results = []
    if delta<=0.998:
        n_upper=500
    else:
        n_upper=3000
    while k_value <= k_upper:
        try:
            if nu0 + k_value == 1:
                Results.append([k_value, 'Null-Fully Responsive'])
            else:
                twoalpha = 2*alpha0 + k_value
                nu = nu0 + k_value
                
                # Calculate Lambda and tau
                try:
                    Lambda = np.sqrt((1 - (1/nu)**2) / twoalpha)
                    tau = (1 - delta/nu)/Lambda
                    
                    # For delta < 1, calculate mu_star for use in convergence check
                    if abs(delta - 1.0) >= 1e-10:  # delta < 1
                        mu_star = 1/((nu0 + k_value)*Lambda)
                    
                    S_0 = 0
                    n = max(n_start, k_value + 2)
                    bound = 0
                    
                    while True:
                        # Compute S0 for current n
                        S0_result = S_c0(mu, sigma, alpha0, nu0, n, delta, beta, G)
                        
                        # Get S0 value - for delta=1, mu_val is ignored
                        try:
                            S0_prev = S_0
                            if abs(delta - 1.0) < 1e-10:  # delta = 1
                                S0_val = get_S0_value(S0_result, k_value, None)
                            else:  # delta < 1
                                S0_val = get_S0_value(S0_result, k_value, mu_star)
                                
                            S_0 = float(S0_val)
                        except Exception as e:
                            print(f"Warning: Error getting S0 value: {e}")
                            S_0 = 0.0
                        
                        # Different convergence checks for delta=1 vs delta<1
                        if delta==1:  # delta = 1
                            pass
                        else:  # delta < 1
                            # For delta<1, check convergence with G-dependent threshold
                            if n > k_value + 2:
                                try:
                                    # For larger G, threshold decreases proportionally
                                    convergence_threshold = 1e-6
                                    
                                    if abs(S_0 - S0_prev) < convergence_threshold or n >= n_upper+1:
                                        bound = 1
                                        print(f"Difference: {abs(S_0 - S0_prev):.6f}")
                                        break
                                except Exception as e:
                                    print(f"Warning: Error in convergence check: {e}")
                        
                        # Check if S0 exceeds tau
                        if S_0 > tau:
                            break
                        n += 1
                    
                    if bound == 1:
                        for k in range(k_value, k_upper + 1):
                            Results.append([k, 'Null'])
                        break
                    
                    n_max = n - 1
                    n_start = n_max
                    Results.append([k_value, n_max])
                    print(f"k={k_value}, n_max={n_max}, S_0={S_0}, tau={tau}")
                    
                except (ValueError, ZeroDivisionError) as e:
                    # Handle calculation errors (like division by zero)
                    print(f"Warning: Calculation error for k={k_value}: {e}")
                    Results.append([k_value, 'Null-Calculation Error'])
        except Exception as e:
            print(f"Error processing k={k_value}: {e}")
            Results.append([k_value, 'Null-Error'])
        
        k_value += 1
    
    # Make sure we have entries for every admissible k value.
    k_values_in_results = [row[0] for row in Results]
    for k in range(k_min, k_upper + 1):
        if k not in k_values_in_results:
            Results.append([k, 'Null-Missing'])
    
    # Sort results by k value
    Results.sort(key=lambda x: x[0])
    
    # Create DataFrame with headers
    df = pd.DataFrame(Results, columns=['k', 'n_max'])
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
    for a tangent root. The returned list contains zero, one, or two offer-value
    thresholds in increasing order.
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
    is_delta_one = abs(delta - 1.0) < 1e-10

    def gain(X):
        innovation = X - mu_prev
        mu_post = mu_prev + innovation/nu
        sigma_post = Lambda*math.sqrt(base_scale_squared + innovation**2)
        scaled_cost = C/sigma_post
        scaled_mu = mu_post/sigma_post
        try:
            S_value = get_S_value(
                S_df, k, None if is_delta_one else scaled_mu, scaled_cost
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return np.nan
        if S_value is None or not np.isfinite(S_value):
            return np.nan
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

    # A double root need not produce a sign change.
    if not roots:
        i_min = int(np.argmin(gain_values))
        if 0 < i_min < len(offer_grid) - 1:
            result = minimize_scalar(
                gain,
                bounds=(offer_grid[i_min - 1], offer_grid[i_min + 1]),
                method="bounded",
                options={"xatol": tol},
            )
            if result.success and abs(result.fun) <= tol:
                roots.append(float(result.x))

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
    """
    Algorithm 4 (derived quantity):
    Compute k**(δ) as the smallest k with S0_infinite(k, μ*, 0) < τ_k for δ<1.
    """
    if delta >= 1.0:
        raise ValueError("This function is only for delta < 1. For delta=1, use a different approach.")
    
    # Compute the infinite-horizon S0 values
    S0_inf_df = S_c0_infinite(
        mu, sigma, alpha0, nu0, delta, beta, G, k_upper=max_k
    )
    
    # Determine k_min based on sigma
    if sigma == 0:
        k_min = max(math.floor(2 - 2 * alpha0), 1)
    else:
        k_min = 1
    
    # Start from k_min
    k = k_min
    
    # For each k, check if S0(k, mu_star) < tau_k
    while k <= max_k:
        # Skip invalid k values where nu0 + k = 1
        if abs(nu0 + k - 1.0) < 1e-10:
            k += 1
            continue
        
        # Calculate parameters for this k
        nu = nu0 + k
        twoalpha = 2 * alpha0 + k
        
        # Calculate Lambda and tau
        Lambda_squared = max(1e-10, (1 - (1/nu)**2) / twoalpha)
        Lambda = np.sqrt(Lambda_squared)
        tau = (1 - delta/nu)/Lambda
        
        # Calculate mu_star
        mu_star = 1/(nu*Lambda)
        
        # Get S0 value for this k
        try:
            S0_val = get_S0_value(S0_inf_df, k, mu_star)
            if S0_val is None:
                break  # k exceeds available values
            
            S0 = float(S0_val.iloc[0] if hasattr(S0_val, 'iloc') else S0_val)
            
            # Check if S0 < tau
            if S0 < tau:
                return k  # Found k**
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Error at k={k}: {e}")
            break
        
        k += 1
    
    # If no k** found up to max_k
    return max_k + 1

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
            print(f"Error in iteration {iterations}: {e}")
            # Be conservative if computation fails
            delta_high = delta_mid
    
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
            print(f"Error in iteration {iterations}: {e}")
            # Be conservative if computation fails
            delta_high = delta_mid
    
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
