import re
import numpy as NP
import scipy.io as SIO
import time
import os
import os.path
import sys
from fastlmm.association.FastLmmSet import FastLmmSet
from fastlmm.association.FastLmmSet import Local
import unittest
import subprocess
import fastlmm.inference.tests.test
import fastlmm.feature_selection.test
import pysnptools.test
import shutil
import logging
import fastlmm.util.util as ut
from pysnptools.util.mapreduce1.distributabletest import DistributableTest
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalInParts
from fastlmm.pyplink.snpreader.Hdf5 import Hdf5  # Need by some *.py's that we eval

tolerance = 1e-4

# def compare_files(afilename, bfilename):
#    if not os.path.isfile(afilename) or not os.path.isfile(bfilename):
#        return False
#    if os.name == 'posix':
#        call = 'wine CompareFiles/CompareFiles.exe'
#    else:
#        call = 'CompareFiles/CompareFiles.exe'
#    out = subprocess.check_output(call+' -tol '+str(tolerance)+' '+afilename+' '
#                                  +bfilename, shell=True)
#    return out.find('Files are comparable.') != -1


class WidgetTestCase(unittest.TestCase):
    def __init__(self, infile):
        unittest.TestCase.__init__(self)

        self._infile = infile
        # if not self._existExpectedOutput():
        #    self._generateExpectedOutput()

    def _tmpOutfile(self):
        outfile = os.path.splitext(self._infile)[0]
        return "tmp/" + outfile + ".txt"

    def _referenceOutfile(self):
        #!!similar code elsewhere
        import platform

        os_string = platform.platform()
        outfile = os.path.splitext(self._infile)[0]

        windows_fn = "expected-Windows/" + outfile + ".txt"
        assert os.path.exists(windows_fn), "Can't find file '{0}'".format(windows_fn)
        debian_fn = "expected-debian/" + outfile + ".txt"
        if not os.path.exists(
            debian_fn
        ):  # If reference file is not in debian folder, look in windows folder
            debian_fn = windows_fn

        if "debian" in os_string or "Linux" in os_string:
            if "Linux" in os_string:
                logging.warning(
                    "comparing to Debian output even though found: %s" % os_string
                )
            return debian_fn
        else:
            if "Windows" not in os_string:
                logging.warning(
                    "comparing to Windows output even though found: %s" % os_string
                )
            return windows_fn

    def runTest(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        tmpOutfile = self._tmpOutfile()
        referenceOutfile = self._referenceOutfile()
        logging.info(f"{'inputs/'+self._infile}")
        with open("inputs/" + self._infile) as f:
            filecontent = f.read()

        runner = Local()
        try:
            distributable = eval(filecontent)
        except Exception as e:
            raise Exception(
                "Can't eval '{0}' because of '{1}'".format(self._infile, e.message)
            )
        runner.run(distributable)

        out, msg = ut.compare_files(tmpOutfile, referenceOutfile, tolerance)
        self.assertTrue(
            out,
            "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile),
        )

    # def _generateExpectedOutput(self):
    #    tmpOutfile = self._tmpOutfile()
    #    referenceOutfile = self._referenceOutfile()
    #    with open('inputs/'+self._infile) as f:
    #        filecontent = f.read()

    #    runner = Local()
    #    exec(filecontent)
    #    runner.run(distributable)

    #    shutil.copyfile(self._tmpOutfile(), self._referenceOutfile())

    def _existExpectedOutput(self):
        return os.path.isfile(self._referenceOutfile())

    def __str__(self):
        return self._infile


##for debuggingk
# def getDebugTestSuite():
#    suite = unittest.TestSuite()
#    suite.addTest(WidgetTestCase('sc_davies_two_kernel_linear_qqfit.N300.needsReorder.py'))
#    return suite


def getTestSuiteX():
    suite = unittest.TestSuite()
    for f in os.listdir("inputs"):
        if re.match(r".*\.py$", f) is None:
            continue
        # from tests.test import WidgetTestCase
        suite.addTest(WidgetTestCase(f))
    return suite


def removeTmpFiles():
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
        # for f in os.listdir( 'tmp' ):
        #    os.remove(os.path.join('tmp', f), )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    removeTmpFiles()

    import fastlmm.association.tests.testepistasis
    import fastlmm.association.tests.test_single_snp
    import fastlmm.association.tests.test_single_snp_linreg
    import fastlmm.association.tests.test_single_snp_select
    import fastlmm.association.tests.test_single_snp_all_plus_select
    import fastlmm.association.tests.test_snp_set
    import fastlmm.association.tests.test_gwas
    import fastlmm.association.tests.test_heritability_spatial_correction
    import fastlmm.association.tests.test_single_snp_scale
    import fastlmm.inference.tests.test_fastlmm_predictor
    import fastlmm.inference.tests.test_linear_regression
    import fastlmm.inference.tests.test
    import fastlmm.util.test
    import fastlmm.pyplink.test

    suites = unittest.TestSuite(
        [
            # cmk
            # for debugging getDebugTestSuite(),
            getTestSuiteX(),
            fastlmm.inference.tests.test_linear_regression.getTestSuite(),
            fastlmm.association.tests.test_single_snp_scale.getTestSuite(),
            fastlmm.pyplink.test.getTestSuite(),
            fastlmm.feature_selection.test.getTestSuite(),
            # this test is deprecated # fastlmm.association.tests.test_single_snp_select.getTestSuite(),
            fastlmm.inference.tests.test_fastlmm_predictor.getTestSuite(),
            fastlmm.association.tests.test_gwas.getTestSuite(),
            fastlmm.association.tests.test_snp_set.getTestSuite(),
            fastlmm.inference.tests.test.getTestSuite(),
            fastlmm.association.tests.testepistasis.getTestSuite(),
            fastlmm.association.tests.test_heritability_spatial_correction.getTestSuite(),
            fastlmm.util.test.getTestSuite(),
            fastlmm.association.tests.test_single_snp_all_plus_select.getTestSuite(),
            fastlmm.inference.tests.test.getTestSuite(),
            fastlmm.association.tests.test_single_snp.getTestSuite(),
            fastlmm.association.tests.test_single_snp_linreg.getTestSuite(),
        ]
    )

    if True:  # Standard test run
        r = unittest.TextTestRunner(failfast=True)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    else:  # Cluster test run
        # Because both pysnsptools and fastlmm contain a tests folder, to run on cluster must have fastlmm listed first in the PYTHONPATH

        runner = Local()
        runner = LocalMultiProc(taskcount=12, mkl_num_threads=5, just_one_process=False)
        # runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        distributable_test = DistributableTest(suites, "temp_test")
        runner.run(distributable_test)

    debian_count = len(os.listdir("expected-debian"))
    if debian_count > 0:
        logging.warning(
            "The tests contain {0} expected-results files that differ between Windows and Debian".format(
                debian_count
            )
        )

    logging.info("done with testing")
