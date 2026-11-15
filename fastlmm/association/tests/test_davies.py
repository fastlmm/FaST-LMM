"""Independent score-tail references and gx2 diagnostic handling."""

import math
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from gx2._helpers import ImhofClipWarning

from fastlmm.association.tests.Sc import Sc


class TestDavies(unittest.TestCase):
    def test_independent_reference(self):
        # Two pairs of equal weights give a sum of independent exponentials.
        for weight in [1e-3, 1e-8]:
            with self.subTest(weight=weight):
                value = Sc.pv_davies_eig(20, np.array([weight, weight, 1, 1]))
                expected = (math.exp(-10) - weight * math.exp(-10 / weight)) / (1 - weight)
                self.assertAlmostEqual(value, expected, delta=1e-7)

    def test_signed_reference(self):
        # The difference of two chi2_2 variables has a Laplace upper tail.
        for statistic in [-4.0, 0.0, 4.0]:
            with self.subTest(statistic=statistic):
                expected = math.exp(-statistic / 2) / 2 if statistic >= 0 else 1 - math.exp(statistic / 2) / 2
                actual = Sc.pv_davies_eig(statistic, np.array([1., 1., -1., -1.]))
                self.assertAlmostEqual(actual, expected, delta=1e-7)

    def test_exact_support_and_legacy_point_mass(self):
        with patch("gx2.cdf") as cdf:
            for coefficients in [[], [0.], [0., 0.]]:
                for statistic, expected in [(-1, 1.), (0, 1.), (1, 0.)]:
                    with self.subTest(coefficients=coefficients, statistic=statistic):
                        self.assertEqual(Sc.pv_davies_eig(statistic, np.array(coefficients)), expected)
            self.assertEqual(Sc.pv_davies_eig(0, np.array([0., 1.])), 1.)
            self.assertEqual(Sc.pv_davies_eig(0, np.array([-1., 0.])), 0.)
            cdf.assert_not_called()

    def test_flagged_tiny_tail_is_independently_bounded(self):
        for scale in [1e-12, 1., 1e12]:
            with self.subTest(scale=scale):
                with self.assertWarns(ImhofClipWarning):
                    actual = Sc.pv_davies_eig(100 * scale, np.array([.001, .001, 1., 1.]) * scale)
                expected = math.exp(-50) / .999
                self.assertAlmostEqual(actual, expected, delta=1e-7)

    def test_flagged_ordinary_tail_is_rejected(self):
        # Historical Davies fault 2 could carry an approximation; it was still
        # a failure to meet accuracy, not an accepted p-value (may no longer apply).
        for value in [0., .25]:
            with self.subTest(value=value):
                with patch("gx2.cdf", return_value=(value, True, None)):
                    with self.assertRaisesRegex(ArithmeticError, "unreliable tail"):
                        Sc.pv_davies_eig(2, np.array([1., 1.]))

    def test_flagged_answer_must_be_inside_bound(self):
        with patch("gx2.cdf", return_value=(.1, True, None)):
            with self.assertRaisesRegex(ArithmeticError, "unreliable tail"):
                Sc.pv_davies_eig(100, np.array([1., 1.]))

    def test_negative_roundoff_coefficients_are_preserved(self):
        coefficients = np.array([-1e-16, .001, .001, 1., 1.])
        with patch("gx2.cdf", return_value=(0., True, None)) as cdf:
            self.assertEqual(Sc.pv_davies_eig(100, coefficients), 0.)
            np.testing.assert_array_equal(cdf.call_args.args[1], coefficients)

    def test_numeric_error_bounds_are_checked(self):
        for diagnostic in [1e-6, -1., float("nan"), float("inf"), "unknown", [False]]:
            with self.subTest(diagnostic=diagnostic):
                with patch("gx2.cdf", return_value=(.25, diagnostic, None)):
                    with self.assertRaises(ArithmeticError):
                        Sc.pv_davies_eig(2, np.array([1., 1.]))
        for diagnostic in [None, False, 0., 1e-8, 1e-7]:
            with self.subTest(diagnostic=diagnostic):
                with patch("gx2.cdf", return_value=(.25, diagnostic, None)):
                    self.assertEqual(Sc.pv_davies_eig(2, np.array([1., 1.])), .25)

    def test_invalid_probabilities_are_rejected(self):
        for probability in [None, float("nan"), float("inf"), -.1, 1.1, -400., [.25], .25j]:
            with self.subTest(probability=probability):
                with patch("gx2.cdf", return_value=(probability, False, None)):
                    with self.assertRaisesRegex(ArithmeticError, "invalid probability"):
                        Sc.pv_davies_eig(2, np.array([1., 1.]))

    def test_invalid_numeric_input_is_rejected(self):
        inputs = [(float("nan"), [1.]), (float("inf"), [1.]), (1., [float("nan")]),
                  (1., [float("inf")]), (1., [[1.]]), ([1.], [1.]), (1j, [1.]),
                  (1., [1j]), ("1", [1.]), (1., ["1"])]
        with patch("gx2.cdf") as cdf:
            for statistic, coefficients in inputs:
                with self.subTest(statistic=statistic, coefficients=coefficients):
                    with self.assertRaisesRegex(ArithmeticError, "finite real"):
                        Sc.pv_davies_eig(statistic, coefficients)
            cdf.assert_not_called()

    def test_binding_exception_is_preserved(self):
        error = TypeError("invalid coefficient type")
        with patch("gx2.cdf", side_effect=error):
            with self.assertRaises(TypeError) as caught:
                Sc.pv_davies_eig(2, np.array([1., 1.]))
            self.assertIs(caught.exception, error)

    def test_threaded_repeatability(self):
        def calculate(statistic):
            return Sc.pv_davies_eig(statistic, np.array([.1, .3, .5, .8]))
        statistics = np.linspace(1, 4, 12)
        expected = list(map(calculate, statistics))
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(list(executor.map(calculate, statistics)), expected)


def getTestSuite():
    return unittest.defaultTestLoader.loadTestsFromTestCase(TestDavies)
