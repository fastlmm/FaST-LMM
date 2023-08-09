from __future__ import absolute_import
from __future__ import print_function
import sys
import numpy as np
import logging
import unittest
import os.path
import doctest
import pandas as pd
import sys

from fastlmm.association import single_snp
from fastlmm.association import single_snp_linreg
import pysnptools.util.pheno as pstpheno
from fastlmm.feature_selection.test import TestFeatureSelection
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
from pysnptools.kernelreader import  Identity as KernelIdentity
from pysnptools.standardizer import Unit
from pysnptools.snpreader import Bed, Pheno, SnpData
from pysnptools.kernelreader import SnpKernel


class TestSingleSnpLinReg(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.bedbase = os.path.join(self.pythonpath, 'tests/datasets/all_chr.maf0.001.N300')
        self.phen_fn = os.path.join(self.pythonpath, 'tests/datasets/phenSynthFrom22.23.N300.randcidorder.txt')
        self.cov_fn = os.path.join(self.pythonpath,  'tests/datasets/all_chr.maf0.001.covariates.N300.txt')

    tempout_dir = "tempout/single_snp_linreg"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_linreg(self):
        logging.info("TestSingleSnpLinReg test_linreg")
        test_snps = Bed(self.bedbase,count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("linreg")

        frame1 = single_snp_linreg(test_snps=test_snps[:,:10], pheno=pheno,
                                    covar=covar, 
                                    output_file_name=output_file,
                                    count_A1=False
                                    )

        frame1 = frame1[['sid_index', 'SNP', 'Chr', 'GenDist', 'ChrPos', 'PValue']]
        self.compare_files(frame1,"linreg")

        frame2 = single_snp_linreg(test_snps=test_snps[:,:10], pheno=pheno, 
                                    covar=covar, 
                                    output_file_name=output_file,count_A1=False
                                    )
        self.compare_files(frame2,"linreg")


    def compare_files(self,frame,ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp/"+ref_base+".txt") #Results are in single_snp, not single_snp_lin_reg

        #sid_list,pvalue_list = frame['SNP'].values,frame['Pvalue'].values

        #sid_to_pvalue = {}
        #for index, sid in enumerate(sid_list):
        #    sid_to_pvalue[sid] = pvalue_list[index]

        reference=pd.read_csv(reffile,delimiter='\s',comment=None,engine='python')
        assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(reffile)
        for _, row in reference.iterrows():
            sid = row.SNP
            pvalue = frame[frame['SNP'] == sid].iloc[0].PValue
            assert abs(row.PValue - pvalue) < 1e-5, "pair {0} differs too much from file '{1}'".format(sid,reffile)

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testmod(sys.modules['fastlmm.association.single_snp_linreg'])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


def getTestSuite():
    

    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpLinReg)
    return unittest.TestSuite([suite1])



if __name__ == '__main__':



    # this import is needed for the runner
    from fastlmm.association.tests.test_single_snp_linreg import TestSingleSnpLinReg
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=False)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    else: #Cluster test run
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest
        runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print(runner.run(distributable_test))


    logging.info("done with testing")
