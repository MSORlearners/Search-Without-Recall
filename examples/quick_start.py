from algorithm import S_fast

if __name__ == "__main__":
    alpha0, nu0 = -0.5, 0.0
    mu_flag, sigma_flag = 0, 0
    beta, G = 1.0, 50
    n, delta = 60, 0.95

    S_df, c_grid, mu_grid = S_fast(mu_flag, sigma_flag, alpha0, nu0, n, delta, beta, G)
    print(S_df.head())
