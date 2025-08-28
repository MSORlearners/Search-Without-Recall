"""
Quick Start Demo for Algorithms 1–4
-----------------------------------

This script demonstrates minimal, fast-running calls to each algorithm group
from the working paper's appendix. It uses small grids to keep runtime low.

Algorithms:
- Algorithm 1: S_fast / S_c0 (finite-horizon recursion)
- Algorithm 2: data_n_min (minimum horizon table)
- Algorithm 3: threshold (decision thresholds)
- Algorithm 4: S_c0_infinite / S_infinite / k_doublestar / delta_star / delta_k_star
"""

from pprint import pprint

from algorithm import (
    S_fast,
    S_c0,
    data_n_min,
    threshold,
    S_c0_infinite,
    S_infinite,
    k_doublestar,
    delta_star,
    delta_k_star,
)


def demo_algorithm_1():
    print("\n=== Algorithm 1: Scaled foresight recursion (finite horizon) ===")
    # Small parameters for a quick run
    alpha0, nu0 = -0.5, 0.0
    mu_flag, sigma_flag = 0, 0
    beta, G = 1.0, 40
    n, delta = 40, 0.9

    # Over (mu, c) when 0 < delta < 1, or over c when delta = 1
    S_df= S_fast(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)[0]
    print("S_fast output (head):")
    print(S_df.head())

    # Also show c=0 specialization (S_c0)
    S0_df, _ = S_c0(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)
    print("\nS_c0 output (head):")
    print(S0_df.head())


def demo_algorithm_2():
    print("\n=== Algorithm 2: Minimum horizon table n*(k) ===")
    # Keep tiny k_upper for a fast demo
    alpha0, nu0 = -0.5, 0.0
    mu_flag, sigma_flag = 0, 0
    beta, G = 1.0, 30
    delta, k_upper = 0.9, 3

    table = data_n_min(mu_flag, sigma_flag, alpha0, nu0, delta, beta, G, k_upper)
    print("n*(k) table:")
    print(table)


def demo_algorithm_3():
    print("\n=== Algorithm 3: Decision thresholds ===")
    # Compute a small S table first (Algorithm 1)
    alpha0, nu0 = -0.5, 0.0
    mu_flag, sigma_flag = 0, 0
    beta, G = 1.0, 40
    n, delta = 40, 0.9
    S_df= S_fast(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)[0]

    # Pick a representative state for the threshold computation
    k = 3
    mu_prev = 0.0
    sigma_prev = 1.0
    C = 1.0  # cost parameter

    roots = threshold(k, mu_prev, sigma_prev, alpha0, nu0, delta, S_df, C)
    print(f"Threshold roots for k={k}:")
    pprint(roots)


def demo_algorithm_4():
    print("\n=== Algorithm 4: Infinite horizon & critical discount factors ===")
    alpha0, nu0 = -0.5, 0.0
    mu_flag, sigma_flag = 0, 0
    beta, G = 1.0, 12
    delta = 0.9
    k_upper = 4

    # Infinite-horizon (c = 0)
    S0_inf_df, mu_grid = S_c0_infinite(mu_flag, sigma_flag, alpha0, nu0, delta, beta, G)
    print("S_c0_infinite (first row):")
    print(S0_inf_df.head(1))

    # Infinite-horizon (general case)
    S_inf_df, _ = S_infinite(mu_flag, sigma_flag, alpha0, nu0, delta, beta, G, k_upper)
    print("\nS_infinite (head):")
    print(S_inf_df.head())

    # k** (first k at which fully responsive policy becomes optimal)
    k2 = k_doublestar(mu_flag, sigma_flag, alpha0, nu0, delta, beta, G, max_k=6)
    print(f"\nk** (approx): {k2}")

    # Critical discount factor at k_min
    delta_star_val = delta_star(mu_flag, sigma_flag, alpha0, nu0, beta, G, tol=1e-4, max_iter=10)
    print(f"delta* (approx, k_min): {delta_star_val}")

    # Critical discount factor at selected k
    k = 2
    delta_k_val = delta_k_star(k, mu_flag, sigma_flag, alpha0, nu0, beta, G, tol=1e-4, max_iter=10)
    print(f"delta*_k (approx, k={k}): {delta_k_val}")


if __name__ == "__main__":
    demo_algorithm_1()
    demo_algorithm_2()
    demo_algorithm_3()
    demo_algorithm_4()
