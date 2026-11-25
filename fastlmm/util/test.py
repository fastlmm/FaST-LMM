import doctest
import sys

import unittest
import os.path


# We do it this way instead of using doctest.DocTestSuite because doctest.DocTestSuite requires modules to be pickled, which python doesn't allow.
# We need tests to be pickleable so that they can be run on a cluster.
class TestDocStrings(unittest.TestCase):

    def test_vertex_cut(self):
        import fastlmm.util.VertexCut

        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        result = doctest.testmod(fastlmm.util.VertexCut)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

    def test_sample_util(self):
        import fastlmm.util.util

        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))


        result = doctest.testmod(fastlmm.util.util)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

    def deprecated_test_compute_auto_pcs(self):

        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        result = doctest.testmod(sys.modules["fastlmm.util.compute_auto_pcs"])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


class TestPlotP(unittest.TestCase):

    def test_qqplot_legend_handles(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        from fastlmm.util.stats import plotp

        figure, axes = plt.subplots()
        try:
            axes.plot([0.0, 1.0], [0.0, 1.0], ".")
            plt.sca(axes)
            plotp.addqqplotinfo(
                np.array([0.0, 1.0]),
                2,
                alphalevel=None,
                legendlist=["observed"],
            )

            legend = axes.get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual(legend.legend_handles[0].get_markersize(), 10)
        finally:
            plt.close(figure)


class TestMinimize1D(unittest.TestCase):
    """minimize1D refines grid minima with SciPy's brent, which re-evaluates the bracket."""

    grid = [index / 10 for index in range(11)]

    def quadratic(self, x):
        return (x - 0.42) ** 2

    def flaky(self, on_reevaluation):
        # Like an inner optimization (findA2 calling findH2), the second evaluation
        # of the grid minimum at 0.4 comes out different from the first one.
        import numpy as np

        seen = set()

        def function(x):
            x = float(x)
            if x == 0.4 and x in seen:
                return on_reevaluation()
            seen.add(x)
            return self.quadratic(x)

        return function, np.array(self.grid)

    def test_rejected_bracket_falls_back_to_bounded_search(self):
        from fastlmm.util.mingrid import minimize1D

        function, grid = self.flaky(lambda: 1.0)  # no longer the lowest of the triplet
        x_min, f_min = minimize1D(function, evalgrid=grid)
        self.assertAlmostEqual(x_min, 0.42, places=6)
        self.assertAlmostEqual(f_min, 0.0, places=10)

    def test_other_errors_from_the_function_propagate(self):
        from fastlmm.util.mingrid import minimize1D

        def fail():
            raise ValueError("inner optimization failed")

        function, grid = self.flaky(fail)
        with self.assertRaisesRegex(ValueError, "inner optimization failed"):
            minimize1D(function, evalgrid=grid)

    def test_valid_bracket_result_is_unchanged(self):
        import numpy as np
        import scipy.optimize as opt
        from fastlmm.util.mingrid import minimize1D

        expected = opt.brent(self.quadratic, brack=(0.3, 0.4, 0.5), full_output=True)
        x_min, f_min = minimize1D(self.quadratic, evalgrid=np.array(self.grid))
        self.assertEqual((x_min, f_min), (expected[0], expected[1]))


def getTestSuite():
    """
    set up composite test suite
    """

    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDocStrings))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlotP))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMinimize1D))

    return test_suite


if __name__ == "__main__":
    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=False)
    ret = r.run(suites)
    assert ret.wasSuccessful()
    print("done")
