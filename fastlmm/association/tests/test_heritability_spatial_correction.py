from __future__ import absolute_import
import numpy as np
import scipy as sp
import logging
import unittest
import os.path
import time
import sys
import doctest
import pandas as pd
import fastlmm.util.util as ut
from fastlmm.association.heritability_spatial_correction import heritability_spatial_correction
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
from pysnptools.snpreader import Dat, Bed, Pheno, SnpData
from fastlmm.feature_selection.test import TestFeatureSelection
from six.moves import range

tolerance = 1e-4


class TestHeritabilitySpatialCorrection(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.snpreader_whole = Bed(self.pythonpath + "/tests/datasets/synth/all.bed",count_A1=False)
        self.pheno_whole = Pheno(self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt")

    tempout_dir = "tempout/heritability_spatial_correction"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name)
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_one(self):
        '''
        Lock in results on arbitrary data -- because meaningful runs take too long to run.
        '''
        fn = "one.txt"
        logging.info(fn)
        tmpOutfile = self.file_name(fn)

        half = self.pheno_whole.read().val
        pheno = SnpData(iid=self.pheno_whole.iid,sid=["pheno0","pheno1"],val=np.c_[half,half])

        spatial_coor = [[i,-i] for i in range(self.snpreader_whole.iid_count)]
        alpha_list = alpha_list_big=[int(v) for v in np.logspace(2,np.log10(4000), 2)]
        dataframe = heritability_spatial_correction(self.snpreader_whole,spatial_coor,self.snpreader_whole.iid,alpha_list,2,pheno,jackknife_count=2,permute_plus_count=1,permute_times_count=1,just_testing=True)

        dataframe.to_csv(tmpOutfile,sep="\t",index=False)
        referenceOutfile = TestFeatureSelection.reference_file("heritability_spatial_correction/"+fn)
        out,msg=ut.compare_files(tmpOutfile, referenceOutfile, tolerance)                
        self.assertTrue(out, "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile))

    def test_two(self):
        '''
        Lock in results on arbitrary data -- because meaningful runs take too long to run.
        '''
        fn = "two.txt"
        logging.info(fn)
        tmpOutfile = self.file_name(fn)

        snpreader = self.snpreader_whole[:10,:]

        spatial_coor = [[i,-i] for i in range(snpreader.iid_count)]
        alpha_list = alpha_list_big=[int(v) for v in np.logspace(2,np.log10(4000), 2)]
        dataframe = heritability_spatial_correction(snpreader,spatial_coor,snpreader.iid,alpha_list,2,self.pheno_whole,jackknife_count=2,permute_plus_count=1,permute_times_count=1,just_testing=False)

        dataframe.to_csv(tmpOutfile,sep="\t",index=False)
        referenceOutfile = TestFeatureSelection.reference_file("heritability_spatial_correction/"+fn)
        out,msg=ut.compare_files(tmpOutfile, referenceOutfile, tolerance)                
        self.assertTrue(out, "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile))

    def test_three(self):
        '''
        Lock in results on arbitrary data -- because meaningful runs take too long to run.
        '''
        fn = "three.txt"
        logging.info(fn)
        tmpOutfile = self.file_name(fn)

        snpreader = self.snpreader_whole[:10,:]

        spatial_coor = [[i,-i] for i in range(snpreader.iid_count)]
        alpha_list = alpha_list_big=[int(v) for v in np.logspace(2,np.log10(4000), 2)]
        dataframe = heritability_spatial_correction(snpreader,spatial_coor,snpreader.iid,alpha_list,2,self.pheno_whole,jackknife_count=0,permute_plus_count=0,permute_times_count=0,just_testing=False)

        dataframe.to_csv(tmpOutfile,sep="\t",index=False)
        referenceOutfile = TestFeatureSelection.reference_file("heritability_spatial_correction/"+fn)
        out,msg=ut.compare_files(tmpOutfile, referenceOutfile, tolerance)                
        self.assertTrue(out, "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile))

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testmod(sys.modules['fastlmm.association.heritability_spatial_correction'])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__



def getTestSuite():
    
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestHeritabilitySpatialCorrection)
    return unittest.TestSuite([suite1])



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # this import is needed for the runner
    #from fastlmm.association.tests.test_heritability_spatical_correction import TestHeritabilitySpatialCorrection
    suites = unittest.TestSuite([getTestSuite()])

    r = unittest.TextTestRunner(failfast=False)
    ret = r.run(suites)
    assert ret.wasSuccessful()
    logging.info("done with testing")
