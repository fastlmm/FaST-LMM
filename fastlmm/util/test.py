import numpy as np
import scipy as sp
import logging
import doctest

import unittest
import os.path
import time


   

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

        import matplotlib.pyplot as plt
        plt.switch_backend('agg') #Needed so can test manhanttan_plot on machine with no DISPLAY

        result = doctest.testmod(fastlmm.util.util)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

    def test_compute_auto_pcs(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        result = doctest.testfile("compute_auto_pcs.py")
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__



def getTestSuite():
    """
    set up composite test suite
    """
    
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDocStrings))

    return test_suite

if __name__ == '__main__':

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=False)
    r.run(suites)
    print "done"
