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


def test_full_support_quadrature_and_paper_resolution():
    import algorithm as alg

    assert alg.PAPER_GRID_SIZE == 400
    _, nodes, _, weights = alg._make_grids(80)
    mass = sum(alg._t_pdf(nodes[j], 2.0) * weights[j] for j in range(1, 81))
    assert abs(mass - 1.0) < 1e-10


def test_jeffreys_tail_regression():
    """The full-support calculation gives n*(1)=6, not the truncated value 7."""
    import algorithm as alg

    values_6, _, _ = alg._S_c0_compute(0, 0, -0.5, 0.0, 6, 1.0, 1.0, 160)
    values_7, _, _ = alg._S_c0_compute(0, 0, -0.5, 0.0, 7, 1.0, 1.0, 160)
    assert values_6[3, 1] < 1.0 < values_7[3, 1]
    assert abs(values_7[3, 1] - 1.034638056) < 1e-4


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
    assert set(table["k"]) == set(range(3, p["k_upper"] + 1))


def test_algo3_threshold():
    import algorithm as alg

    p = _tiny_params()
    S_df, _, _ = alg.S_fast(
        p["mu_flag"], p["sigma_flag"], p["alpha0"], p["nu0"], p["n"], p["delta"], p["beta"], p["G"]
    )

    # Choose a modest state; we only assert that it returns a list (possibly empty)
    k = 3
    mu_prev = 0.0
    sigma_prev = 1.0
    C = 1.0

    roots = alg.threshold(k, mu_prev, sigma_prev, p["alpha0"], p["nu0"], p["delta"], S_df, C)
    assert isinstance(roots, list)

    assert roots == sorted(roots)
    assert all(math.isfinite(root) for root in roots)


def test_algo3_sign_changing_and_tangent_roots(monkeypatch):
    """Algorithm 3 returns offer thresholds and detects a double root."""
    import algorithm as alg

    nu, delta, cost = 3.0, 0.9, 1.0

    def install_gain(target):
        def fake_get_S_value(_S_df, _k, scaled_mu, scaled_cost):
            sigma_post = cost/scaled_cost
            mu_post = scaled_mu*sigma_post
            offer = nu*mu_post
            return (scaled_cost/cost)*(
                target(offer) + cost - delta*mu_post + offer
            )
        monkeypatch.setattr(alg, "get_S_value", fake_get_S_value)

    install_gain(lambda offer: offer)
    roots = alg.threshold(3, 0.0, 1.0, -0.5, 0.0, delta, None, cost)
    assert roots == pytest.approx([0.0], abs=1e-12)

    install_gain(lambda offer: (offer + 1.0)*(offer - 1.0))
    roots = alg.threshold(3, 0.0, 1.0, -0.5, 0.0, delta, None, cost)
    assert roots == pytest.approx([-1.0, 1.0], abs=1e-7)

    install_gain(lambda offer: (offer - 0.137)**2)
    roots = alg.threshold(3, 0.0, 1.0, -0.5, 0.0, delta, None, cost)
    assert roots == pytest.approx([0.137], abs=1e-6)


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

    dk = alg.delta_k_star(3, mu_flag, sigma_flag, alpha0, nu0, beta, G, tol=1e-4, max_iter=8)
    assert isinstance(dk, float) and 0.0 < dk < 1.0
