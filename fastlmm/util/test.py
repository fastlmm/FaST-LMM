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


def getTestSuite():
    """
    set up composite test suite
    """

    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDocStrings))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlotP))

    return test_suite


if __name__ == "__main__":
    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=False)
    ret = r.run(suites)
    assert ret.wasSuccessful()
    print("done")
