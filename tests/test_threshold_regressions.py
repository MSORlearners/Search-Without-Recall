"""Regression tests for horizon equality and convex stopping-region searches."""

import contextlib
import io
import math
import unittest
from unittest.mock import patch

import numpy as np

import algorithm as alg


class ThresholdRegressionTests(unittest.TestCase):
    def roots_for_gain(self, target, offers):
        """Inject a convex gain, retaining threshold's state update and root search."""
        nu, delta, cost = 3.0, 0.9, 1.0

        def foresight(_table, _k, scaled_mu, scaled_cost):
            sigma = cost / scaled_cost
            mu = scaled_mu * sigma
            offer = nu * mu
            return (target(offer) + cost - delta * mu + offer) / sigma

        # At this state the offer-grid transformation is x = tan(theta).
        with patch.object(alg, "get_S_value", side_effect=foresight), patch.object(
            alg.np, "linspace", return_value=np.arctan(np.asarray(offers, dtype=float))
        ):
            return alg.threshold(
                3, 0.0, 1.0, -0.5, 0.0, delta, None, cost,
                tol=1e-10, grid_size=len(offers),
            )

    def assert_roots(self, actual, expected):
        self.assertEqual(len(actual), len(expected), actual)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

    def test_two_roots_in_one_unsampled_interval(self):
        roots = self.roots_for_gain(lambda x: (x - 0.5) ** 2 - 0.01, [-2, 0, 1, 2])
        self.assert_roots(roots, [0.4, 0.6])

    def test_minimum_next_to_grid_boundary(self):
        for center in (0.25, 1.75):
            with self.subTest(center=center):
                roots = self.roots_for_gain(
                    lambda x: (x - center) ** 2 - 0.01, [0, 1, 2]
                )
                self.assert_roots(roots, [center - 0.1, center + 0.1])

    def test_tangent_root(self):
        roots = self.roots_for_gain(lambda x: (x - 0.5) ** 2, [-2, 0, 1, 2])
        self.assert_roots(roots, [0.5])

    def test_positive_minimum_has_no_roots(self):
        self.assertEqual(
            self.roots_for_gain(lambda x: (x - 0.5) ** 2 + 0.01, [-2, 0, 1, 2]), []
        )

    def test_sign_changing_roots(self):
        roots = self.roots_for_gain(lambda x: x * x - 1, [-2, 0, 2])
        self.assert_roots(roots, [-1, 1])

    def test_grid_roots_are_not_duplicated(self):
        roots = self.roots_for_gain(lambda x: x * x - 1, [-2, -1, 0, 1, 2])
        self.assert_roots(roots, [-1, 1])

    def test_decreasing_gain_has_one_root(self):
        roots = self.roots_for_gain(lambda x: 0.5 - x, [-2, 0, 2])
        self.assert_roots(roots, [0.5])

    def test_horizon_equality_is_retained(self):
        tau = (1 - 1 / 2) / math.sqrt((1 - 1 / 4) / 3)
        horizons = []

        def compute(_mu, _sigma, _alpha, _nu, n, _delta, _beta, _grid):
            horizons.append(n)
            return n

        with patch.object(alg, "S_c0", side_effect=compute), patch.object(
            alg, "get_S0_value", side_effect=lambda n, _k, _mu: tau if n == 3 else tau + 0.1
        ), contextlib.redirect_stdout(io.StringIO()):
            result = alg.data_n_min(0, 1, 1, 1, 1, 1, 4, 1)

        self.assertEqual(horizons, [3, 4])
        self.assertEqual(result.loc[0, "n_max"], 3)


if __name__ == "__main__":
    unittest.main()
