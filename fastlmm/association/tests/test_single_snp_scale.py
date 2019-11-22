import doctest
import unittest
import numpy as np
import logging
import os
import pandas as pd
import tempfile
import shutil

import pysnptools.util as pstutil
from pysnptools.snpreader import Bed, DistributedBed
from pysnptools.util.filecache import LocalCache
from pysnptools.util.filecache import PeerToPeer
from pysnptools.snpreader import SnpGen

from fastlmm.association import single_snp
from fastlmm.association import single_snp_scale
from pysnptools.util.mapreduce1.runner import LocalMultiProc

class TestSingleSnpScale(unittest.TestCase):
    @classmethod

    def test_snpgen(self):
        seed = 0
        snpgen = SnpGen(seed=seed,iid_count=1000,sid_count=5000)
        snpdata = snpgen[:,[0,1,200,2200,10]].read()
        np.testing.assert_allclose(np.nanmean(snpdata.val,axis=0),np.array([ 0.00253807,  0.00127877,  0.16644993,  0.00131406,  0.00529101]),rtol=1e-5)

        snpdata2 = snpgen[:,[0,1,200,2200,10]].read()
        np.testing.assert_equal(snpdata.val,snpdata2.val)
        snpdata3 = snpgen[::10,[0,1,200,2200,10]].read()
        np.testing.assert_equal(snpdata3.val,snpdata2.val[::10,:])

    def test_snpgen_cache(self):
        cache_file = tempfile.gettempdir() + "/test_snpgen_cache.snpgen.npz"
        if os.path.exists(cache_file):
            os.remove(cache_file)
        snpgen = SnpGen(seed=0,iid_count=1000,sid_count=5000,cache_file=cache_file)
        assert os.path.exists(cache_file)
        snpgen2 = SnpGen(seed=0,iid_count=1000,sid_count=5000,cache_file=cache_file)
        os.remove(cache_file)
        snpdata = snpgen2[:,[0,1,200,2200,10]].read()
        np.testing.assert_allclose(np.nanmean(snpdata.val,axis=0),np.array([ 0.00253807,  0.00127877,  0.16644993,  0.00131406,  0.00529101]),rtol=1e-5)

    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        import fastlmm as fastlmm
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(fastlmm.__file__)),".."))
        self.bed = Bed(os.path.join(self.pythonpath, 'tests/datasets/synth/all'),count_A1=True)[:,::10]
        self.phen_fn = os.path.join(self.pythonpath, 'tests/datasets/synth/pheno_10_causals.txt')
        self.cov_fn = os.path.join(self.pythonpath,  'tests/datasets/synth/cov.txt')

    tempout_dir = "tempout/single_snp_scale"

    def test_old(self):
        logging.info("test_old")

        output_file = self.file_name("old")
        results_df = single_snp(self.bed,  self.phen_fn , covar=self.cov_fn, count_A1=True,
                                  output_file_name=output_file
                                  )
        self.compare_files(results_df,"old")

    @staticmethod
    def _cache_dict(top, clear_cache):
        storage = top.join("intermediate")
        if clear_cache:
            storage.rmtree()
        cache_dict={chrom:storage for chrom in xrange(23)}
        return cache_dict
        
    def test_low(self):
        logging.info("test_low")

        output_file = self.file_name("low")

        storage = LocalCache("local_cache/low")
        for clear_cache in (True, False):
            if clear_cache:
                storage.rmtree()
            results_df = single_snp_scale(test_snps=self.bed, pheno=self.phen_fn, covar=self.cov_fn, cache=storage, output_file_name=output_file)
            self.compare_files(results_df,"old")

    def test_local_distribute(self):
        logging.info("test_local_distribute")
        force_python_only = False

        output_file = self.file_name("local_distribute")

        storage = LocalCache("local_cache/local_distribute")
        test_storage = storage.join('test_snps')
        test_storage.rmtree('')
        test_snps = DistributedBed.write(test_storage, self.bed, piece_per_chrom_count=2)

        results_df = single_snp_scale(test_snps=test_snps, pheno=self.phen_fn, covar=self.cov_fn, G0=self.bed,cache=self._cache_dict(storage,clear_cache=True),
                                    output_file_name=output_file, force_python_only=force_python_only
                                    )

        self.compare_files(results_df,"old")

        results_df = single_snp_scale(test_snps=self.bed, pheno=self.phen_fn, covar=self.cov_fn, G0=self.bed,cache=self._cache_dict(storage,clear_cache=False), 
                                    output_file_name=output_file
                                    )
        self.compare_files(results_df,"old")

    def test_mapreduce1_runner(self):
        logging.info("test_mapreduce1_runner")

        output_file = self.file_name("mapreduce1_runner")
        runner = LocalMultiProc(taskcount=4,just_one_process=True)
        results_df = single_snp_scale(test_snps=self.bed, pheno=self.phen_fn,covar=self.cov_fn,output_file_name=output_file,runner=runner)
        self.compare_files(results_df,"old")



    def test_old_one(self):
        logging.info("test_old_one")

        output_file = self.file_name("old_one")

        test_snps3 = self.bed[:,self.bed.pos[:,0]==3] # Test only on chromosome 3
        results_df = single_snp(test_snps=test_snps3, K0=self.bed,  pheno=self.phen_fn , covar=self.cov_fn, count_A1=True,
                                  output_file_name=output_file,
                                  )
        self.compare_files(results_df,"old_one")


    def test_one_fast(self):
        logging.info("test_one_fast")

        output_file = self.file_name("one_fast")

        storage = LocalCache("local_cache")
        test_storage = storage.join('one_fast')
        test_storage.rmtree()
        test_snps3 = self.bed[:,self.bed.pos[:,0]==3] # Test only on chromosome 3
        test_snps3_dist = DistributedBed.write(test_storage,test_snps3,piece_per_chrom_count=2)

        results_df = single_snp_scale(test_snps=test_snps3_dist, pheno=self.phen_fn, covar=self.cov_fn, G0=self.bed, output_file_name=output_file)
        self.compare_files(results_df,"old_one")
    
    def test_one_chrom(self):
        logging.info("test_one_chrom")

        output_file = self.file_name("one_chrom")

        storage = LocalCache("local_cache/one_chrom")
        test_storage = storage.join('test_snps')
        test_storage.rmtree('')
        test_snps3 = self.bed[:,self.bed.pos[:,0]==3] # Test only on chromosome 3
        test_snps3_dist = DistributedBed.write(test_storage,test_snps3,piece_per_chrom_count=2)


        for test_snps, ref, clear_cache, name in (
                                                    (test_snps3, "old_one", True, "Run with just chrom3"),
                                                    (test_snps3_dist, "old_one", True, "Run with distributed test SNPs"),
                                                    (test_snps3, "old_one", False, "Run with just chrom3 (use cache)"),
                                                    (test_snps3_dist, "old_one", False, "Run with distributed test SNPs (use cache)"),
                                                    ):
            logging.info("=========== " + name + " ===========")
            results_df = single_snp_scale(test_snps=test_snps, pheno=self.phen_fn, covar=self.cov_fn, K0=self.bed,cache=self._cache_dict(storage,clear_cache=clear_cache), 
                                     output_file_name=output_file,
                                      )
            self.compare_files(results_df,ref)


    def test_peertopeer(self):
        logging.info("test_peertopeer")

        output_file = self.file_name("peertopeer")

        def id_and_path_function():
             from pysnptools.util.filecache import ip_address_pid
             ip_pid = ip_address_pid()
             #Need to put the 'cache_top' here explicitly.
             return ip_pid, 'peertopeer/{0}'.format(ip_pid)
        storage = PeerToPeer(common_directory='peertopeer/common',id_and_path_function=id_and_path_function)
        test_snps_cache = storage.join('test_snps')
        test_snps_cache.rmtree()
        test_snps = DistributedBed.write(test_snps_cache, self.bed, piece_per_chrom_count=2)

        runner = LocalMultiProc(taskcount=5) #Run on 5 additional Python processes

        for clear_cache in (True, False):
            if clear_cache:
                storage.join('cache').rmtree()
            results_df = single_snp_scale(test_snps=test_snps, pheno=self.phen_fn, covar=self.cov_fn, cache=storage.join('cache'), output_file_name=output_file, runner=runner)
            self.compare_files(results_df,"old")


    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def compare_files(self,frame,ref_base):
        reffile = self.reference_file("single_snp_scale/"+ref_base+".txt")

        reference=pd.read_csv(reffile,delimiter='\s',comment=None,engine='python')
        assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(reffile)
        for _, row in reference.iterrows():
            sid = row.SNP
            pvalue = frame[frame['SNP'] == sid].iloc[0].PValue
            assert abs(row.PValue - pvalue) < 1e-5, "pair {0} differs too much from file '{1}'".format(sid,reffile)

    @staticmethod
    def reference_file(outfile):
        #!!similar code elsewhere
        import platform;
        os_string=platform.platform()

        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','..',"tests")
        windows_fn = file_path + '/expected-Windows/'+outfile
        assert os.path.exists(windows_fn), "Can't find file '{0}'".format(windows_fn)
        debian_fn = file_path + '/expected-debian/'+outfile
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

    def Xtest_doctest(self): #Can't get doc test to work so marked out.
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testfile("../single_snp_scale.py",optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__
        

def getTestSuite():
    """
    set up composite test suite
    """
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpScale))   #Tests Ludicrous Speed GWAS
    return test_suite


if __name__ == '__main__':
    #from fastlmm.ludicrous.test import getTestSuite, TestOneMil
    logging.basicConfig(level=logging.INFO)
    suites = getTestSuite()


    if True:
        r = unittest.TextTestRunner(failfast=True)
        r.run(suites)
    else: #runner test run
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest
        runner = LocalMultiProc(taskcount=22,mkl_num_threads=5,just_one_process=False)
        distributable_test = DistributableTest(suites,"temp_test")
        print runner.run(distributable_test)


    logging.info("done")
