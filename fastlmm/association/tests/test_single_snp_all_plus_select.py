import pandas as pd
import sys
import numpy as np
import logging
import unittest
import doctest
import os.path
from pysnptools.snpreader import Bed, Pheno,SnpData
from fastlmm.association import single_snp_all_plus_select,single_snp
from fastlmm.feature_selection.test import TestFeatureSelection
import multiprocessing
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalInParts, LocalMultiThread

class TestSingleSnpAllPlusSelect(unittest.TestCase): 

    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.bedbase = os.path.join(self.pythonpath, 'tests/datasets/all_chr.maf0.001.N300')
        self.phen_fn = os.path.join(self.pythonpath, 'tests/datasets/phenSynthFrom22.23.N300.randcidorder.txt')
        self.cov_fn = os.path.join(self.pythonpath,  'tests/datasets/all_chr.maf0.001.covariates.N300.txt')

    tempout_dir = "tempout/single_snp_all_plus_select"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def too_slow_test_notebook(self):
        do_plot = False
        runner = LocalMultiProc(multiprocessing.cpu_count(),mkl_num_threads=2)
        output_file_name = self.file_name("notebook")


        logging.info("TestSingleSnpAllPlusSelect test_notebook")
        # define file names
        snp_reader = Bed(self.pythonpath + "/tests/datasets/synth/all.bed",count_A1=False)
        pheno_fn = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"
        cov_fn = self.pythonpath + "/tests/datasets/synth/cov.txt"

        # find the chr5 SNPs
        test_snps = snp_reader[:,snp_reader.pos[:,0] == 5]

        #select the 2nd kernel and run GWAS
        results = single_snp_all_plus_select(test_snps=test_snps,G=snp_reader,pheno=pheno_fn,GB_goal=2,do_plot=do_plot,output_file_name=output_file_name,runner=runner, count_A1=False)


        self.compare_files(results,"notebook")

    def test_one(self):
        logging.info("TestSingleSnpAllPlusSelect test_one")
        snps = self.bedbase
        pheno = self.phen_fn
        covar = self.cov_fn
        
        output_file_name = self.file_name("one")
        results = single_snp_all_plus_select(test_snps=snps, pheno=pheno,
                                  covar=covar,
                                  k_list = np.logspace(start=0, stop=1, num=2, base=2),
                                  n_folds=2,
                                  do_plot=False,
                                  output_file_name = output_file_name,
                                  GB_goal=2,
                                  #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5,just_one_process=True)
                                  count_A1=False
                                  )

        self.compare_files(results,"one")

    def too_slow_test_three(self):
        logging.info("TestSingleSnpAllPlusSelect test_three")

        bed_fn = self.pythonpath + "/tests/datasets/synth/all.bed"
        bed_fn = Bed(bed_fn,count_A1=False)
        pheno_fn = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"
        cov_fn = self.pythonpath + "/tests/datasets/synth/cov.txt"
        runner = LocalMultiProc(multiprocessing.cpu_count(),mkl_num_threads=2)

        output_file_name = self.file_name("three")
        results = single_snp_all_plus_select(test_snps=bed_fn, pheno=pheno_fn,
                                  covar=cov_fn,
                                  k_list = [int(k) for k in np.logspace(0, 7, base=2, num=7)],
                                  n_folds=7,
                                  seed = 42,
                                  do_plot=False,
                                  GB_goal=2,
                                  output_file_name=output_file_name,
                                  runner = runner,count_A1=False
                                  )
        logging.info(results)
        self.compare_files(results,"three")

    def too_slowtest_two(self): #!!! rather a big test case
        from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
        logging.info("TestSingleSnpAllPlusSelect test_two")
        do_plot = False

        bed_fn = self.pythonpath + "/tests/datasets/synth/all.bed"
        pheno_fn = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"
        cov_fn = self.pythonpath + "/tests/datasets/synth/cov.txt"

        # partition snps on chr5 vs rest
        test_chr = 5
        snp_reader = Bed(bed_fn,count_A1=False)
        test_snps = snp_reader[:,snp_reader.pos[:,0] == test_chr]
        runner = LocalMultiProc(multiprocessing.cpu_count(),mkl_num_threads=2)

        output_file_name = self.file_name("two")
        for GB_goal in [None,2]:
            results = single_snp_all_plus_select(test_snps=test_snps, G=bed_fn, pheno=pheno_fn,
                                      covar=cov_fn,
                                      k_list = [int(k) for k in np.logspace(0, 7, base=2, num=7)],
                                      n_folds=7,
                                      seed = 42,
                                      do_plot=do_plot,
                                      GB_goal=GB_goal,
                                      output_file_name=output_file_name,
                                      runner = runner,
                                      count_A1=False
                                      )
            logging.info(results.head())
            self.compare_files(results,"two")

    def test_old(self):
        do_plot = False
        from fastlmm.feature_selection.feature_selection_two_kernel import FeatureSelectionInSample
        from pysnptools.util import intersect_apply

        logging.info("TestSingleSnpAllPlusSelect test_old")

        bed_fn = self.pythonpath + "/tests/datasets/synth/all.bed"
        pheno_fn = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"
        cov_fn = self.pythonpath + "/tests/datasets/synth/cov.txt"

        #load data
        ###################################################################
        snp_reader = Bed(bed_fn,count_A1=False)
        pheno = Pheno(pheno_fn)
        cov = Pheno(cov_fn)

        # intersect sample ids
        snp_reader, pheno, cov = intersect_apply([snp_reader, pheno, cov])

        # read in snps

        # partition snps on chr5 vs rest
        test_chr = 5
        G0 = snp_reader[:,snp_reader.pos[:,0] != test_chr].read(order='C').standardize()
        test_snps = snp_reader[:,snp_reader.pos[:,0] == test_chr].read(order='C').standardize()


        y = pheno.read().val[:,0]
        y -= y.mean()
        y /= y.std()

        # load covariates
        X_cov = cov.read().val
        X_cov.flags.writeable = False

        # invoke feature selection to learn which SNPs to use to build G1
        logging.info("running feature selection conditioned on background kernel")
        # partition data into the first 50 SNPs on chr1 and all but chr1

        select = FeatureSelectionInSample(max_log_k=7, n_folds=7, order_by_lmm=True, measure="ll", random_state=42)
        best_k, feat_idx, best_mix, best_delta = select.run_select(G0.val, G0.val, y, cov=X_cov)    

        # plot out of sample error
        if do_plot: select.plot_results(measure="ll")
        # select.plot_results(measure="mse")

        # print results
        logging.info("best_k:{0}".format(best_k))
        logging.info("best_mix:{0}".format(best_mix))
        logging.info("best_delta:{0}".format(best_delta))


        ###############################
        # use selected SNPs to build G1
        logging.info(feat_idx)
        G1 = G0[:,feat_idx]

        output_file_name = self.file_name("old")
        results_df = single_snp(test_snps, pheno, G0=G0, G1=G1, mixing=best_mix, h2=None,leave_out_one_chrom=False,output_file_name=output_file_name,count_A1=False)

        logging.info("results:")
        logging.info("#"*40)
        logging.info(results_df.head())
        self.compare_files(results_df,"old")


    def compare_files(self,frame,ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp_all_plus_select/"+ref_base+".txt")

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

    def too_slow_test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testmod(sys.modules['fastlmm.association.single_snp_all_plus_select'])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__



def getTestSuite():
    
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpAllPlusSelect)
    return unittest.TestSuite([suite1])



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # this import is needed for the runner
    from fastlmm.association.tests.test_single_snp_all_plus_select import TestSingleSnpAllPlusSelect
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=True)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    else: #Cluster test run



        from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest


        runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        distributable_test = DistributableTest(suites,"temp_test")
        print(runner.run(distributable_test))


    logging.info("done with testing")
