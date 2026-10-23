"""Regression tests for grid lookup and boundary classification."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import algorithm as alg


class LookupRegressionTests(unittest.TestCase):
    def test_mean_first_point_and_first_interval(self):
        grid = np.array([-2., -1., 0., 1.])
        table = pd.DataFrame([[4., 1., 0., 1.]], index=['k=3'],
                             columns=[f'mu={x:.6f}' for x in grid])
        for source in (table, (table, grid)):
            for mean, expected in [(-2, 4), (-1.5, 2.5), (-3, 7), (2, 2)]:
                with self.subTest(mean=mean, tuple_input=isinstance(source, tuple)):
                    self.assertEqual(alg.get_S0_value(source, 3, mean), expected)

    def test_stage_matching_is_exact(self):
        table = pd.DataFrame([[99., 99.], [2., 0.]],
                             index=['n=40, k_min=3, k=30', 'n=40, k_min=3, k=3'],
                             columns=['c=1.0', 'c=0.0'])
        self.assertEqual(alg.get_S_value(table, 3, None, 0), 0)
        self.assertEqual(alg.get_S_value(table, 3, None, 2), 4)
        with self.assertRaises(KeyError):
            alg.get_S_value(table, 4, None, 0)

    def test_legacy_duplicate_labels_are_rejected(self):
        table = pd.DataFrame([[1., 2.]], index=['n=5, k_min=3, k=3'],
                             columns=['c=0.0', 'c=0.0'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            alg.get_S_value(table, 3, None, 0)

    def test_discounted_full_precision_and_bilinear_lookup(self):
        # G=400 formerly produced only 117 distinct cost labels per stage.
        G = 400
        c, _, means, _ = alg._make_grids(G)
        raw = np.zeros((5, G + 2, G + 2))
        raw[3] = 3 + 2 * means[:, None] + 4 * c[None, :]
        with patch.object(alg, '_S_fast_compute', return_value=(raw, 3, c, means)):
            table, _, _ = alg.S_fast(0, 0, -.5, 0, 5, .9, 1, G)
        self.assertTrue(table.index.is_unique)
        self.assertTrue(table.columns.is_unique)
        for mean, cost in [(means[1], 0), (-31, 1e-20), (.37, .03), (31, 400)]:
            self.assertAlmostEqual(alg.get_S_value(table, 3, mean, cost),
                                   3 + 2 * mean + 4 * cost, places=10)

    def test_zero_cost_matches_raw_kernel_and_exact_one_step(self):
        G = 400
        raw, kmin, costs, means = alg._S_fast_compute_delta1(0, 0, -.5, 0, 5, 1, 1, G)
        with patch.object(alg, '_S_fast_compute', return_value=(raw, kmin, costs, means)):
            table, _, _ = alg.S_fast(0, 0, -.5, 0, 5, 1, 1, G)
        self.assertEqual(len(set(table.columns)), G)
        result = alg.get_S_value(table, 3, None, 0)
        self.assertEqual(result, raw[3, G])
        self.assertAlmostEqual(result, 3 / (4 * math.sqrt(2)), delta=2e-6)

    def test_discount_arbitrarily_close_to_one_uses_discounted_shape(self):
        with patch.object(alg, '_S_fast_compute_delta_less1', return_value='discounted') as kernel:
            result = alg._S_fast_compute(0, 0, -.5, 0, 5, 1 - 1e-12, 1, 4)
        self.assertEqual(result, 'discounted')
        kernel.assert_called_once()

    def test_discounted_zero_cost_agrees_with_separate_solver(self):
        full, _, grid = alg.S_fast(0, 0, -.5, 0, 6, .9, 1, 32)
        zero = alg.S_c0(0, 0, -.5, 0, 6, .9, 1, 32)
        for k in (3, 4, 5):
            for mean in (grid[1], .5, grid[32]):
                self.assertAlmostEqual(alg.get_S_value(full, k, mean, 0),
                                       alg.get_S0_value(zero, k, mean), places=12)

    def test_threshold_does_not_hide_invalid_table(self):
        table = pd.DataFrame([[1., 2.]], index=['n=5, k_min=3, k=3'],
                             columns=['c=0.0', 'c=0.0'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            alg.threshold(3, 0, 1, -.5, 0, 1, table, 0)

    def test_threshold_runs_on_corrected_table(self):
        table = alg.S_fast(0, 0, -.5, 0, 24, .9, 1, 16)[0]
        roots = alg.threshold(3, 0, 1, -.5, 0, .9, table, 1)
        self.assertEqual(roots, sorted(roots))
        self.assertTrue(all(math.isfinite(root) for root in roots))

    def test_small_infinite_horizon_smoke(self):
        result = alg.k_doublestar(0, 0, -.5, 0, .9, 1, 16, max_k=6)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 3)


class BoundaryRegressionTests(unittest.TestCase):
    @staticmethod
    def tau(k, alpha=1, nu0=1, delta=.9):
        return (nu0 + k - delta) * math.sqrt((2 * alpha + k) / ((nu0 + k) ** 2 - 1))

    def test_crossing_precedes_stabilization(self):
        tau = self.tau(1)
        with patch.object(alg, 'S_c0', side_effect=lambda *args: args[4]), patch.object(
                alg, 'get_S0_value', side_effect=lambda n, *_: tau + (1e-7 if n == 4 else -1e-7)):
            result = alg.data_n_min(0, 0, 1, 1, .9, 1, 4, 1, n_upper=4)
        self.assertEqual(result.loc[0, 'n_max'], 3)
        self.assertEqual(result.loc[0, 'status'], 'crossing-found')

    def test_cap_and_stabilization_do_not_claim_infinity_or_exact_boundary(self):
        with patch.object(alg, 'S_c0'), patch.object(alg, 'get_S0_value', return_value=0):
            capped = alg.data_n_min(0, 0, 1, 1, 1, 1, 4, 1, n_upper=4)
            stable = alg.data_n_min(0, 0, 1, 1, .9, 1, 4, 1, n_upper=4)
        for result, status in [(capped, 'horizon-cap'), (stable, 'numerical-stabilization')]:
            self.assertTrue(pd.isna(result.loc[0, 'n_max']))
            self.assertEqual(result.loc[0, 'status'], status)
            self.assertEqual(result.loc[0, 'n_checked'], 4)

    def test_horizon_initial_reversal(self):
        def value(n, k, _mean):
            return 10 if k == 2 and n == 4 else 0
        with patch.object(alg, 'S_c0', side_effect=lambda *args: args[4]), patch.object(
                alg, 'get_S0_value', side_effect=value):
            result = alg.data_n_min(0, 0, .5, .5, .9, 1, 4, 1, n_upper=5)
        self.assertEqual(result.loc[0, 'n_max'], 3)

    def test_k_doublestar_includes_equality(self):
        with patch.object(alg, 'S_c0_infinite'), patch.object(
                alg, 'get_S0_value', side_effect=lambda _table, k, _mean: self.tau(k, -.5, 0)):
            self.assertEqual(alg.k_doublestar(0, 0, -.5, 0, .9, 1, 4, max_k=4), 3)

    def test_k_doublestar_initial_reversal(self):
        with patch.object(alg, 'S_c0_infinite'), patch.object(
                alg, 'get_S0_value', side_effect=lambda _table, k, _mean: 10 if k == 2 else 0):
            self.assertEqual(alg.k_doublestar(0, 0, .5, .5, .9, 1, 4, max_k=3), 3)

    def test_k_doublestar_no_crossing_is_unresolved(self):
        with patch.object(alg, 'S_c0_infinite'), patch.object(alg, 'get_S0_value', return_value=100):
            self.assertIsNone(alg.k_doublestar(0, 0, -.5, 0, .9, 1, 4, max_k=4))

    def test_failed_discount_search_does_not_invent_boundary(self):
        with patch.object(alg, 'S_c0_infinite', side_effect=RuntimeError('not converged')):
            with self.assertRaisesRegex(RuntimeError, 'no boundary was inferred'):
                alg.delta_star(0, 0, -.5, 0, 1, 4)
            with self.assertRaisesRegex(RuntimeError, 'no boundary was inferred'):
                alg.delta_k_star(3, 0, 0, -.5, 0, 1, 4)


if __name__ == '__main__':
    unittest.main()
