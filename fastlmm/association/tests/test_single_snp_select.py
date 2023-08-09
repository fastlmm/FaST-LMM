from __future__ import absolute_import
from __future__ import print_function
#import matplotlib
#matplotlib.use("TKAgg",warn=False)
#import pylab
import pandas as pd
import sys

import numpy as np
import logging
import unittest
import doctest
import os.path
from pysnptools.snpreader import Bed, Pheno,SnpData
from fastlmm.association import single_snp_select
from fastlmm.feature_selection.test import TestFeatureSelection
import platform
import multiprocessing
from pysnptools.util.mapreduce1.runner import LocalMultiProc


class TestSingleSnpSelect(unittest.TestCase): 

    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.bedbase = os.path.join(self.pythonpath, 'tests/datasets/all_chr.maf0.001.N300')
        self.phen_fn = os.path.join(self.pythonpath, 'tests/datasets/phenSynthFrom22.23.N300.randcidorder.txt')
        self.cov_fn = os.path.join(self.pythonpath,  'tests/datasets/all_chr.maf0.001.covariates.N300.txt')

    tempout_dir = "tempout/single_snp_select"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    #Break these tests up into six parts so they can run faster when on cluster.
    def test_sel_plus_pc_h2Search(self): #!!! rather a big test case
        logging.info("TestSingleSnpSelect sel_plus_pc_h2Search")
        self._sel_plus_pc(None,None,None,count_A1=False)

    def test_sel_plus_pc_h2Search_low(self): #!!! rather a big test case
        logging.info("TestSingleSnpSelect sel_plus_pc_h2Search_low")
        self._sel_plus_pc(None,True,False,count_A1=False)

    def test_sel_plus_pc_h2Search_full(self): #!!! rather a big test case
        logging.info("TestSingleSnpSelect sel_plus_pc_h2Search_full")
        self._sel_plus_pc(None,False,True,count_A1=False)

    def test_sel_plus_pc_h2IsHalf(self): #!!! rather a big test case
        logging.info("TestSingleSnpSelect sel_plus_pc_h2IsHalf")
        self._sel_plus_pc(.5,None,None,count_A1=False)

    def test_sel_plus_pc_h2IsHalf_low(self): #!!! rather a big test case
        logging.info("TestSingleSnpSelect sel_plus_pc_h2IsHalf_low")
        self._sel_plus_pc(.5,True,False,count_A1=False)

    def test_sel_plus_pc_h2IsHalf_full(self): #!!! rather a big test case
        logging.info("TestSingleSnpSelect sel_plus_pc_h2IsHalf_full")
        self._sel_plus_pc(.5,False,True,count_A1=False)

    def _sel_plus_pc(self,h2,force_low_rank,force_full_rank,count_A1=None):
        do_plot = False
        use_cache = False

        # define file names
        bed_fn = self.pythonpath + "/tests/datasets/synth/all.bed"
        phen_fn = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"

        pcs_fn = os.path.join(self.tempout_dir,"sel_plus_pc.pcs.txt")
        if not (use_cache and os.path.exists(pcs_fn)):
            from fastlmm.util import compute_auto_pcs
            covar = compute_auto_pcs(bed_fn,count_A1=count_A1)
            logging.info("selected number of PCs: {0}".format(covar["vals"].shape[1]))
            Pheno.write(pcs_fn,SnpData(iid=covar['iid'],sid=covar['header'],val=covar['vals']))
        else:
            logging.info("Using top pcs's cache")
            covar=Pheno(pcs_fn)

        runner = LocalMultiProc(multiprocessing.cpu_count(),mkl_num_threads=2)

        logging.info("Working on h2={0},force_low_rank={1},force_full_rank={2}".format(h2,force_low_rank,force_full_rank))
        result_file_name = "sel_plus_pc_{0}".format("h2IsHalf" if h2 == .5 else "h2Search")
        output_file_name = os.path.join(self.tempout_dir,result_file_name)+".txt"
        results = single_snp_select(test_snps=bed_fn, G=bed_fn, pheno=phen_fn,
                                        k_list = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,125,160,200,250,320,400,500,630,800,1000],
                                        h2=h2,
                                        n_folds = self.pythonpath + "/tests/datasets/synth/DebugEmitFolds.txt",
                                        covar=covar,
                                        output_file_name=output_file_name,
                                        force_low_rank=force_low_rank,force_full_rank=force_full_rank,
                                        GB_goal=2,
                                        count_A1=False
                                        #runner = runner
                                    )
        logging.info(results.head())
        self.compare_files(results,result_file_name)


    def test_old_sel_plus_pc(self): #!!! rather a big test case
        if platform.system() == "Darwin": #Don't run old C code on Mac
            return

        logging.info("TestSingleSnpSelect old_sel_plus_pc")

        from pysnptools.snpreader import Bed
        from fastlmm.util import compute_auto_pcs

        # define file names
        bed_fn = self.pythonpath + "/tests/datasets/synth/all"
        cov_fn = "pcs_cov.txt"

        # run PCgeno
        #TODO: rename to auto_pcs
        result = compute_auto_pcs(bed_fn, output_file_name=cov_fn,count_A1=False)
        logging.info("selected number of PCs: {0}".format(result["vals"].shape[1]))

        # import algorithms
        from fastlmm.util.run_fastlmmc import runFASTLMM, runLMMSELECT

        # set some file paths for fastlmmc
        phen_fn = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"
        out_dir = self.tempout_dir
        fastlmm_path = self.pythonpath + "/external/fastlmmc"

        # consists of two fastlmmc calls, one that does feature selection and one that runs GWAS
        for suffix,logdelta in [("h2IsHalf",0),("h2Search",None)]:
            result_file_name = "sel_plus_pc_old_{0}".format("h2IsHalf" if logdelta == 0 else "h2Search")
            runLMMSELECT(bed_fn, phen_fn, out_dir, result_file_name, bfileSim=bed_fn, covar=cov_fn, fastlmm_path=fastlmm_path,autoSelectCriterionMSE=False,excludeByGeneticDistance=1000,optLogdelta=logdelta)
            # compare sel_plus_pc_old_h2*.LMMSELECT.out.txt
            short = result_file_name+".LMMSELECT.out"
            results=pd.read_csv(self.tempout_dir+"/"+short+".txt",delimiter='\s',comment=None,engine='python')
            results['PValue']=results.Pvalue #add a new column with different capitalization
            self.compare_files(results,short)



    def compare_files(self,frame,ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp_select/"+ref_base+".txt")

        #sid_list,pvalue_list = frame['SNP'].values,frame['Pvalue'].values

        #sid_to_pvalue = {}
        #for index, sid in enumerate(sid_list):
        #    sid_to_pvalue[sid] = pvalue_list[index]

        reference=pd.read_csv(reffile,delimiter='\s',comment=None,engine='python')
        if 'Pvalue' in reference.columns: reference['PValue']=reference.Pvalue #add a new column with different capitalization if it is there


        assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(reffile)
        for _, row in reference.iterrows():
            sid = row.SNP
            pvalue = frame[frame['SNP'] == sid].iloc[0].PValue
            assert abs(row.PValue - pvalue) < 1e-5, "pair {0} differs too much from file '{1}'".format(sid,reffile)

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testmod(sys.modules['fastlmm.association.single_snp_select'])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__



def getTestSuite():
    
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpSelect)
    return unittest.TestSuite([suite1])



if __name__ == '__main__':

    # this import is needed for the runner
    from fastlmm.association.tests.test_single_snp_select import TestSingleSnpSelect
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=False)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    else: #Cluster test run



        from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest


        #runner = HPC(10, 'RR1-N13-09-H44',r'\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\Redmond',
        #                remote_python_parent=r"\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\REDMOND\carlk\Source\carlk\july_7_14\tests\runs\2014-07-24_15_02_02_554725991686\pythonpath",
        #                update_remote_python_parent=True,
        #                priority="AboveNormal",mkl_num_threads=1)
        runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print(runner.run(distributable_test))


    logging.info("done with testing")
