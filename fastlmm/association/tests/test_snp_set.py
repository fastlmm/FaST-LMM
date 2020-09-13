from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import scipy as sp
import logging
import unittest
import os.path
import time
import sys
import doctest
import pandas as pd

from fastlmm.association import snp_set
import fastlmm.pyplink.plink as plink
from fastlmm.feature_selection.test import TestFeatureSelection
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalInParts
import fastlmm.util.util as ut

tolerance = 1e-4


class TestSnpSet(unittest.TestCase):

    #!!created a Expect Durbin, too

    @classmethod
    def setUpClass(self):

        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.currentFolder = os.path.dirname(os.path.realpath(__file__))

    tempout_dir = "tempout/snp_set"
 
    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name)
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_one(self):
        logging.info("TestSnpSet test_one")

        fn = "lrt_one_kernel_fixed_mixed_effect_linear_qqfit.N300.txt"
        tmpOutfile = self.file_name(fn)
        referenceOutfile = self._referenceOutfile(fn)

        result_dataframe = snp_set(
            test_snps = self.currentFolder+'/../../../tests/datasets/all_chr.maf0.001.N300',
            set_list = self.currentFolder+'/../../../tests/datasets/set_input.23.txt',
            pheno = self.currentFolder+'/../../../tests/datasets/phenSynthFrom22.23.N300.txt',
            output_file_name = tmpOutfile
            )


        out,msg=ut.compare_files(tmpOutfile, referenceOutfile, tolerance)                
        self.assertTrue(out, "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile))

    def test_two(self):
        logging.info("TestSnpSet test_two")

        fn = "lrt_up_two_kernel_mixed_effect_linear_qqfit.N300.fullrank.txt"
        tmpOutfile = self.file_name(fn)
        referenceOutfile = self._referenceOutfile(fn)

        result_dataframe = snp_set(
            test_snps = self.currentFolder+'/../../../tests/datasets/all_chr.maf0.001.N300',
            set_list = self.currentFolder+'/../../../tests/datasets/set_input.23.txt',
            pheno = self.currentFolder+'/../../../tests/datasets/phenSynthFrom22.23.N300.txt',
            G0 = self.currentFolder+'/../../../tests/datasets/all_chr.maf0.001.chr22.23.N300.bed',
            output_file_name = tmpOutfile,
            test="lrt"
            )


        out,msg=ut.compare_files(tmpOutfile, referenceOutfile, tolerance)                
        self.assertTrue(out,msg)#msg='Files %s and %s are different.' % (tmpOutfile, referenceOutfile))


    def test_three(self):
        logging.info("TestSnpSet test_three")

        fn = "sc_davies_one_kernel_linear_qqfit.N300.txt"
        tmpOutfile = self.file_name(fn)
        referenceOutfile = self._referenceOutfile(fn)

        result_dataframe = snp_set(
            test_snps = self.currentFolder+'/../../../tests/datasets/all_chr.maf0.001.N300',
            set_list = self.currentFolder+'/../../../tests/datasets/set_input.small.txt',
            pheno = self.currentFolder+'/../../../tests/datasets/phenSynthFrom22.23.N300.txt',
            test = "sc_davies",
            output_file_name = tmpOutfile
            )


        out,msg=ut.compare_files(tmpOutfile, referenceOutfile, tolerance)                
        self.assertTrue(out, "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile))


    def test_four(self):
        logging.info("TestSnpSet test_four")

        fn = "sc_davies_two_kernel_linear_qqfit.N300.noautoselect.txt"
        tmpOutfile = self.file_name(fn)
        referenceOutfile = self._referenceOutfile(fn)

        result_dataframe = snp_set(
            test_snps = self.currentFolder+'/../../../tests/datasets/all_chr.maf0.001.N300',
            set_list = self.currentFolder+'/../../../tests/datasets/set_input.small.txt',
            pheno = self.currentFolder+'/../../../tests/datasets/phenSynthFrom22.23.N300.randcidorder.txt',
            G0 = self.currentFolder+'/../../../tests/datasets/all_chr.maf0.001.chr22.23.N300.bed',
            test = 'sc_davies',
            output_file_name = tmpOutfile
            )


        out,msg=ut.compare_files(tmpOutfile, referenceOutfile, tolerance)                
        self.assertTrue(out, "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile))

    def test_doctest(self):
        result = doctest.testmod(sys.modules['fastlmm.association.snp_set'])
        assert result.failed == 0, "failed doc test: " + __file__

    def _referenceOutfile(self,_infile):
        import platform;
        os_string=platform.platform()
        outfile = os.path.splitext(_infile)[0]

        windows_fn = self.currentFolder+'/../../../tests/expected-Windows/'+outfile+'.txt'
        assert os.path.exists(windows_fn), "Can't find file '{0}'".format(windows_fn)
        debian_fn = self.currentFolder+'/../../../tests/expected-debian/'+outfile  +'.txt'
        if not os.path.exists(debian_fn): #If reference file is not in debian folder, look in windows folder
            debian_fn = windows_fn

        if "debian" in os_string or "Linux" in os_string:
            if "Linux" in os_string:
                logging.warning("comparing to Debian output even though found: %s" % os_string)
            return debian_fn
        else:
            if "Windows" not in os_string:
                logging.warning("comparing to Windows output even though found: %s" % os_string)
            return windows_fn 


    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testmod(sys.modules['fastlmm.association.snp_set'])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

def getTestSuite():
    

    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSnpSet)
    return unittest.TestSuite([suite1])



if __name__ == '__main__':    
    from fastlmm.association.tests.test_snp_set import TestSnpSet
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=False)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    else: #Cluster test run
        from pysnptools.util.mapreduce1.distributabletest import DistributableTest


        runner = HPC(10, 'RR1-N13-09-H44',r'\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\Redmond',
                     remote_python_parent=r"\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\REDMOND\carlk\Source\carlk\july_7_14\tests\runs\2014-07-24_15_02_02_554725991686\pythonpath",
                     update_remote_python_parent=True,
                     priority="AboveNormal",mkl_num_threads=1)
        runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print(runner.run(distributable_test))


    logging.info("done with testing")
