"""
Test Suite (smoke-level) for Algorithms 1–4

These tests aim to be quick. Algorithm 4 can be slow for large grids; we mark it
as 'slow' and skip by default unless RUN_SLOW=1 is set in the environment.

Run:
  pytest -q
Enable slow tests:
  RUN_SLOW=1 pytest -q
"""

import os
import math

import pytest


def _tiny_params():
    # Smallest reasonable settings for a fast pass
    return dict(
        alpha0=-0.5,
        nu0=0.0,
        mu_flag=0,
        sigma_flag=0,
        beta=1.0,
        G=16,       # keep tiny for speed
        n=24,
        delta=0.9,
        k_upper=3,
    )


def test_algo1_s_fast_and_sc0():
    import algorithm as alg

    p = _tiny_params()
    S_df, c_grid, mu_grid = alg.S_fast(
        p["mu_flag"], p["sigma_flag"], p["alpha0"], p["nu0"], p["n"], p["delta"], p["beta"], p["G"]
    )
    assert S_df is not None and not S_df.empty

    S0_df, _ = alg.S_c0(
        p["mu_flag"], p["sigma_flag"], p["alpha0"], p["nu0"], p["n"], p["delta"], p["beta"], p["G"]
    )
    assert S0_df is not None and not S0_df.empty


def test_algo2_data_n_min():
    import algorithm as alg

    p = _tiny_params()
    table = alg.data_n_min(
        p["mu_flag"], p["sigma_flag"], p["alpha0"], p["nu0"], p["delta"], p["beta"], p["G"], p["k_upper"]
    )
    assert table is not None and not table.empty
    # Expect at least rows for k in [1..k_upper]
    assert set(table["k"]).issuperset(set(range(1, p["k_upper"] + 1)))


def test_algo3_threshold():
    import algorithm as alg

    p = _tiny_params()
    S_df, _, _ = alg.S_fast(
        p["mu_flag"], p["sigma_flag"], p["alpha0"], p["nu0"], p["n"], p["delta"], p["beta"], p["G"]
    )

    # Choose a modest state; we only assert that it returns a list (possibly empty)
    k = 2
    mu_prev = 0.0
    sigma_prev = 1.0
    C = 1.0

    roots = alg.threshold(k, mu_prev, sigma_prev, p["alpha0"], p["nu0"], p["delta"], S_df, C)
    assert isinstance(roots, list)

    # If roots exist, they should be within [-tau, tau] by construction
    nu = p["nu0"] + k
    twoalpha = 2 * p["alpha0"] + k
    if nu > 0 and twoalpha > 0:
        Lambda_sq = max(1e-10, (1 - (1 / nu) ** 2) / twoalpha)
        tau = (1 - 1 / nu) / math.sqrt(Lambda_sq)
        for r in roots:
            assert -tau - 1e-6 <= r <= tau + 1e-6


@pytest.mark.skipif(os.environ.get("RUN_SLOW", "0") != "1", reason="Set RUN_SLOW=1 to run slow tests")
def test_algo4_infinite_and_criticals_slow():
    import algorithm as alg

    # Smaller grid still, to keep this "slow" test reasonable
    alpha0, nu0 = -0.5, 0.0
    mu_flag, sigma_flag = 0, 0
    beta, G = 1.0, 12
    delta = 0.9
    k_upper = 3

    S0_inf_df, mu_grid = alg.S_c0_infinite(mu_flag, sigma_flag, alpha0, nu0, delta, beta, G)
    assert S0_inf_df is not None and not S0_inf_df.empty

    S_inf_df, _ = alg.S_infinite(mu_flag, sigma_flag, alpha0, nu0, delta, beta, G, k_upper)
    assert S_inf_df is not None and not S_inf_df.empty

    k2 = alg.k_doublestar(mu_flag, sigma_flag, alpha0, nu0, delta, beta, G, max_k=6)
    assert isinstance(k2, int) and k2 >= 1

    d_star = alg.delta_star(mu_flag, sigma_flag, alpha0, nu0, beta, G, tol=1e-4, max_iter=8)
    assert isinstance(d_star, float) and 0.0 < d_star < 1.0

    dk = alg.delta_k_star(2, mu_flag, sigma_flag, alpha0, nu0, beta, G, tol=1e-4, max_iter=8)
    assert isinstance(dk, float) and 0.0 < dk < 1.0
