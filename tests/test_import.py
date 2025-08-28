def test_import_and_run():
    import algorithm
    # quick smoke test (small G for speed)
    S_df, _, _ = algorithm.S_fast(0, 0, -0.5, 0.0, 20, 0.9, 1.0, 20)
    assert not S_df.empty
