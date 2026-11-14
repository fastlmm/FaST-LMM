"""Reference probabilities and fault propagation for the Davies backend."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from fastlmm.association.tests.Sc import Sc


class TestDavies(unittest.TestCase):
    def test_independent_reference(self):
        # Two pairs of equal weights give a sum of independent exponentials.
        for weight in [1e-3, 1e-8]:
            with self.subTest(weight=weight):
                value = Sc.pv_davies_eig(20, np.array([weight, weight, 1, 1]))
                expected = (math.exp(-10) - weight * math.exp(-10 / weight)) / (1 - weight)
                self.assertAlmostEqual(value, expected, delta=1e-7)

    def test_faults_are_rejected(self):
        trace = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        for fault in [1, 2, 3, 4]:
            with self.subTest(fault=fault):
                # Fault 2 can carry an approximation; it is still a failure
                # to meet the requested accuracy, not an accepted p-value.
                value = 0.25 if fault == 2 else None
                with patch("fastlmmrslib.qf", return_value=(value, fault, trace)):
                    with self.assertRaisesRegex(ArithmeticError, f"fault {fault}") as caught:
                        Sc.pv_davies_eig(2, np.array([1.0, 1.0]))
                    self.assertIn(str(trace), str(caught.exception))

    def test_missing_probability_is_rejected(self):
        with patch("fastlmmrslib.qf", return_value=(None, 0, [0.0] * 7)):
            with self.assertRaisesRegex(ArithmeticError, "fault 0"):
                Sc.pv_davies_eig(2, np.array([1.0, 1.0]))

    def test_invalid_numeric_input_is_rejected(self):
        with self.assertRaisesRegex(ArithmeticError, "fault 3"):
            Sc.pv_davies_eig(float("nan"), np.array([1.0, 1.0]))

    def test_binding_exception_is_preserved(self):
        error = TypeError("invalid coefficient type")
        with patch("fastlmmrslib.qf", side_effect=error):
            with self.assertRaises(TypeError) as caught:
                Sc.pv_davies_eig(2, np.array([1.0, 1.0]))
            self.assertIs(caught.exception, error)


def getTestSuite():
    return unittest.defaultTestLoader.loadTestsFromTestCase(TestDavies)
