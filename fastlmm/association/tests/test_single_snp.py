from __future__ import absolute_import
from __future__ import print_function
import sys
import numpy as np
import logging
import unittest
import os.path
import doctest
import pandas as pd

from fastlmm.association import single_snp
from fastlmm.association import single_snp_linreg
import pysnptools.util.pheno as pstpheno
from fastlmm.feature_selection.test import TestFeatureSelection
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
from pysnptools.kernelreader import  Identity as KernelIdentity
from pysnptools.standardizer import Unit
from pysnptools.snpreader import Bed, Pheno, SnpData
from pysnptools.kernelreader import SnpKernel
from six.moves import range

class TestSingleSnp(unittest.TestCase):

    #!!created a Expect Durbin, too

    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.bedbase = os.path.join(self.pythonpath, 'tests/datasets/all_chr.maf0.001.N300')
        self.phen_fn = os.path.join(self.pythonpath, 'tests/datasets/phenSynthFrom22.23.N300.randcidorder.txt')
        self.cov_fn = os.path.join(self.pythonpath,  'tests/datasets/all_chr.maf0.001.covariates.N300.txt')

    tempout_dir = "tempout/single_snp"

    def test_match_cpp(self):
        '''
        match
            FaSTLMM.207\Data\DemoData>..\.cd.\bin\windows\cpp_mkl\fastlmmc -bfile snps -extract topsnps.txt -bfileSim snps -extractSim ASout.snps.txt -pheno pheno.txt -covar covariate.txt -out topsnps.singlesnp.txt -logDelta 0 -verbose 100

        '''
        logging.info("TestSingleSnp test_match_cpp")
        snps = Bed(os.path.join(self.pythonpath, "tests/datasets/selecttest/snps.bed"), count_A1=False)
        pheno = os.path.join(self.pythonpath, "tests/datasets/selecttest/pheno.txt")
        covar = os.path.join(self.pythonpath, "tests/datasets/selecttest/covariate.txt")
        sim_sid = ["snp26250_m0_.19m1_.19","snp82500_m0_.28m1_.28","snp63751_m0_.23m1_.23","snp48753_m0_.4m1_.4","snp45001_m0_.26m1_.26","snp52500_m0_.05m1_.05","snp75002_m0_.39m1_.39","snp41253_m0_.07m1_.07","snp11253_m0_.2m1_.2","snp86250_m0_.33m1_.33","snp3753_m0_.23m1_.23","snp75003_m0_.32m1_.32","snp30002_m0_.25m1_.25","snp26252_m0_.19m1_.19","snp67501_m0_.15m1_.15","snp63750_m0_.28m1_.28","snp30001_m0_.28m1_.28","snp52502_m0_.35m1_.35","snp33752_m0_.31m1_.31","snp37503_m0_.37m1_.37","snp15002_m0_.11m1_.11","snp3751_m0_.34m1_.34","snp7502_m0_.18m1_.18","snp52503_m0_.3m1_.3","snp30000_m0_.39m1_.39","isnp4457_m0_.11m1_.11","isnp23145_m0_.2m1_.2","snp60001_m0_.39m1_.39","snp33753_m0_.16m1_.16","isnp60813_m0_.2m1_.2","snp82502_m0_.34m1_.34","snp11252_m0_.13m1_.13"]
        sim_idx = snps.sid_to_index(sim_sid)
        test_sid = ["snp26250_m0_.19m1_.19","snp63751_m0_.23m1_.23","snp82500_m0_.28m1_.28","snp48753_m0_.4m1_.4","snp45001_m0_.26m1_.26","snp52500_m0_.05m1_.05","snp75002_m0_.39m1_.39","snp41253_m0_.07m1_.07","snp86250_m0_.33m1_.33","snp15002_m0_.11m1_.11","snp33752_m0_.31m1_.31","snp26252_m0_.19m1_.19","snp30001_m0_.28m1_.28","snp11253_m0_.2m1_.2","snp67501_m0_.15m1_.15","snp3753_m0_.23m1_.23","snp52502_m0_.35m1_.35","snp30000_m0_.39m1_.39","snp30002_m0_.25m1_.25"]
        test_idx = snps.sid_to_index(test_sid)

        for G0,G1 in [(snps[:,sim_idx],KernelIdentity(snps.iid)),(KernelIdentity(snps.iid),snps[:,sim_idx])]:
            frame_h2 = single_snp(test_snps=snps[:,test_idx], pheno=pheno, G0=G0,G1=G1, covar=covar,h2=.5,leave_out_one_chrom=False,count_A1=False)
            frame_log_delta = single_snp(test_snps=snps[:,test_idx], pheno=pheno, G0=G0,G1=G1, covar=covar,log_delta=0,leave_out_one_chrom=False,count_A1=False)
            for frame in [frame_h2, frame_log_delta]:
                referenceOutfile = TestFeatureSelection.reference_file("single_snp/topsnps.single.txt")
                reference = pd.read_csv(referenceOutfile,sep="\t") # We've manually remove all comments and blank lines from this file
                assert len(frame) == len(reference)
                for _, row in reference.iterrows():
                    sid = row.SNP
                    pvalue = frame[frame['SNP'] == sid].iloc[0].PValue
                    reldiff = abs(row.Pvalue - pvalue)/row.Pvalue
                    assert reldiff < .035, "'{0}' pvalue_list differ too much {4} -- {2} vs {3}".format(sid,None,row.Pvalue,pvalue,reldiff)
 
    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_mixing(self):
        logging.info("TestSingleSnp test_mixing")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("mixing")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], leave_out_one_chrom=False,
                                      covar=covar, G1=test_snps[:,100:200],mixing=None,
                                      output_file_name=output_file_name,count_A1=False
                                      )

        self.compare_files(frame,"mixing")

    def test_mixingKs(self):
        logging.info("TestSingleSnp test_mixingKs")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("mixingKs")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,K0=SnpKernel(test_snps[:,10:100],Unit()),leave_out_one_chrom=False,
                                      covar=covar, K1=SnpKernel(test_snps[:,100:200],Unit()),mixing=None,
                                      output_file_name=output_file_name,count_A1=False
                                      )

        self.compare_files(frame,"mixing")


    def test_mixid(self):
        logging.info("TestSingleSnp test_mixid")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("mixid")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], leave_out_one_chrom=False,
                                      covar=covar, K1=KernelIdentity(test_snps.iid),mixing=.25,
                                      output_file_name=output_file_name,count_A1=False
                                      )

        self.compare_files(frame,"mixid")


    def test_one(self):
        logging.info("TestSingleSnp test_one")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("one")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
                                  G0=test_snps, covar=covar, 
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"one")

    def test_linreg(self):
        logging.info("TestSingleSnp test_linreg")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("linreg")

        frame1 = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
                                    G0=KernelIdentity(iid=test_snps.iid), covar=covar, 
                                    output_file_name=output_file,count_A1=False
                                    )

        frame1 = frame1[['sid_index', 'SNP', 'Chr', 'GenDist', 'ChrPos', 'PValue']]
        self.compare_files(frame1,"linreg")

        frame2 = single_snp_linreg(test_snps=test_snps[:,:10], pheno=pheno, 
                                    covar=covar, 
                                    output_file_name=output_file
                                    )
        self.compare_files(frame2,"linreg")

    def test_noK0(self):
        logging.info("TestSingleSnp test_noK0")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("noK0")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=1,leave_out_one_chrom=False,
                                  G1=test_snps, covar=covar, 
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"one")


    def test_gb_goal(self):
        logging.info("TestSingleSnp test_gb_goal")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("gb_goal")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
                                  G0=test_snps, covar=covar, GB_goal=0,
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"one")

        output_file = self.file_name("gb_goal2")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
                                  G0=test_snps, covar=covar, GB_goal=.12,
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"one")

    def test_other(self):
        logging.info("TestSingleSnp test_other")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("other")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,leave_out_one_chrom=False,
                                  K1=test_snps, covar=covar, 
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"one")

    def test_none(self):
        logging.info("TestSingleSnp test_none")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("none")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
                                  K0=KernelIdentity(test_snps.iid), covar=covar, 
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"none")

    def test_interact(self):
        logging.info("TestSingleSnp test_interact")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("interact")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
                                  G0=test_snps, covar=covar, interact_with_snp=1,
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"interact")

    def test_preload_files(self):
        logging.info("TestSingleSnp test_preload_files")
        test_snps = self.bedbase
        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
        covar = pstpheno.loadPhen(self.cov_fn)
        bed = Bed(test_snps, count_A1=False)

        output_file_name = self.file_name("preload_files")

        frame = single_snp(test_snps=bed[:,:10], pheno=pheno, G0=test_snps, mixing=0,leave_out_one_chrom=False,
                                  covar=covar, output_file_name=output_file_name,count_A1=False
                                  )
        self.compare_files(frame,"one")
        
    def test_SNC(self):
        logging.info("TestSNC")
        test_snps = self.bedbase
        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
        covar = pstpheno.loadPhen(self.cov_fn)
        bed = Bed(test_snps, count_A1=False)
        snc = bed.read()
        snc.val[:,2] = [0] * snc.iid_count # make SNP #2 have constant values (aka a SNC)

        output_file_name = self.file_name("snc")

        frame = single_snp(test_snps=snc[:,:10], pheno=pheno, G0=snc, mixing=0,leave_out_one_chrom=False,
                                  covar=covar, output_file_name=output_file_name,count_A1=False
                                  )
        self.compare_files(frame,"snc")

    def test_G0_has_reader(self):
        logging.info("TestSingleSnp test_G0_has_reader")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G0_has_reader")

        frame0 = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, leave_out_one_chrom=False,
                                  covar=covar, mixing=0,
                                  output_file_name=output_file_name,count_A1=False
                                  )
        self.compare_files(frame0,"one")

        frame1 = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=KernelIdentity(test_snps.iid), G1=test_snps, leave_out_one_chrom=False,
                                  covar=covar, mixing=1,
                                  output_file_name=output_file_name,count_A1=False
                                  )
        self.compare_files(frame1,"one")

    def test_no_cov(self):
        logging.info("TestSingleSnp test_no_cov")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn

        output_file_name = self.file_name("no_cov")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, mixing=0,leave_out_one_chrom=False,
                                          output_file_name=output_file_name,count_A1=False
                                          )

        self.compare_files(frame,"no_cov")

    def test_no_cov_b(self):
        logging.info("TestSingleSnp test_no_cov_b")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn

        output_file_name = self.file_name("no_cov_b")
        covar = pstpheno.loadPhen(self.cov_fn)
        covar['vals'] = np.delete(covar['vals'], np.s_[:],1) #Remove all the columns
        covar['header'] = []

        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, leave_out_one_chrom=False,
                                  covar=covar, mixing=0,
                                  output_file_name=output_file_name,count_A1=False
                                  )

        self.compare_files(frame,"no_cov")

    def test_G1(self):
        logging.info("TestSingleSnp test_G1")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1")
        for force_full_rank,force_low_rank in [(False,True),(False,False),(True,False)]:
            logging.info("{0},{1}".format(force_full_rank,force_low_rank))
            frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], leave_out_one_chrom=False,
                                          covar=covar, G1=test_snps[:,100:200],
                                          mixing=.5,force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                          output_file_name=output_file_name,count_A1=False
                                          )
            self.compare_files(frame,"G1")


    def test_file_cache(self):
        logging.info("TestSingleSnp test_file_cache")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1")
        cache_file = self.file_name("cache_file")+".npz"
        if os.path.exists(cache_file):
            os.remove(cache_file)
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], leave_out_one_chrom=False,
                                      covar=covar, G1=test_snps[:,100:200],
                                      mixing=.5,
                                      output_file_name=output_file_name,
                                      cache_file = cache_file,count_A1=False
                                      )
        self.compare_files(frame,"G1")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], leave_out_one_chrom=False,
                                      covar=covar, G1=test_snps[:,100:200],
                                      mixing=.5,
                                      output_file_name=output_file_name,
                                      cache_file = cache_file,count_A1=False
                                      )
        self.compare_files(frame,"G1")


    def test_G1_mixing(self):
        logging.info("TestSingleSnp test_G1_mixing")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1_mixing")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, leave_out_one_chrom=False,
                                      covar=covar, 
                                      G1=test_snps[:,100:200],
                                      mixing=0,
                                      output_file_name=output_file_name,count_A1=False
                                      )

        self.compare_files(frame,"one")

    def test_unknown_sid(self):
        logging.info("TestSingleSnp test_unknown_sid")

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        try:
            frame = single_snp(test_snps=test_snps,G0=test_snps,pheno=pheno,leave_out_one_chrom=False,mixing=0,covar=covar,sid_list=['1_4','bogus sid','1_9'],count_A1=False)
            failed = False
        except:
            failed = True

        assert(failed)

    def test_cid_intersect(self):
        logging.info("TestSingleSnp test_cid_intersect")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
        pheno['iid'] = np.vstack([pheno['iid'][::-1],[['Bogus','Bogus']]])
        pheno['vals'] = np.hstack([pheno['vals'][::-1],[-34343]])

        
        covar = self.cov_fn
        output_file_name = self.file_name("cid_intersect")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, leave_out_one_chrom=False,
                                  covar=covar, mixing=0,
                                  output_file_name=output_file_name,count_A1=False
                                  )

        self.compare_files(frame,"one")

    def compare_files(self,frame,ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp/"+ref_base+".txt")

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
        result = doctest.testmod(sys.modules['fastlmm.association.single_snp'])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

class TestSingleSnpLeaveOutOneChrom(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.bedbase = os.path.join(self.pythonpath, 'fastlmm/feature_selection/examples/toydata.5chrom.bed')
        self.phen_fn = os.path.join(self.pythonpath, 'fastlmm/feature_selection/examples/toydata.phe')
        self.cov_fn = os.path.join(self.pythonpath,  'fastlmm/feature_selection/examples/toydata.cov')

    tempout_dir = "tempout/single_snp"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_leave_one_out_with_prekernels(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_leave_one_out_with_prekernels")
        from pysnptools.kernelstandardizer import DiagKtoN
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        chrom_to_kernel = {}
        for chrom in np.unique(test_snps.pos[:,0]):
            other_snps = test_snps[:,test_snps.pos[:,0]!=chrom]
            kernel = other_snps.read_kernel(standardizer=Unit(),block_size=500) #Create a kernel from the SNPs not used in testing
            chrom_to_kernel[chrom] = kernel.standardize(DiagKtoN()) #improves the kernel numerically by making its diagonal sum to iid_count

        output_file = self.file_name("one_looc_prekernel")
        frame = single_snp(test_snps, pheno,
                                  covar=covar,
                                  K0 = chrom_to_kernel,
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"one_looc")

    def test_one_looc(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_one_looc")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("one_looc")
        frame = single_snp(test_snps, pheno,
                                  covar=covar, mixing=0,
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"one_looc")

    def test_two_looc(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_two_looc")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("two_looc")
        frame = single_snp(test_snps[:,::10], pheno,
                                  covar=covar,
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"two_looc")


    def test_interact_looc(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_interact_looc")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("interact_looc")
        frame = single_snp(test_snps, pheno,
                                  covar=covar, mixing=0, interact_with_snp=0,
                                  output_file_name=output_file,count_A1=False
                                  )

        self.compare_files(frame,"interact_looc")

    def test_covar_by_chrom(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_covar_by_chrom")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = Pheno(self.cov_fn).read()
        covar = SnpData(iid=covar.iid,sid=["pheno-1"],val=covar.val)
        covar_by_chrom = {chrom:self.cov_fn for chrom in range(1,6)}
        output_file = self.file_name("covar_by_chrom")
        frame = single_snp(test_snps, pheno,
                                    covar=covar, mixing=0,
                                    covar_by_chrom=covar_by_chrom,
                                    output_file_name=output_file,count_A1=False
                                    )

        self.compare_files(frame,"covar_by_chrom")

    def test_covar_by_chrom_mixing(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_covar_by_chrom_mixing")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn
        covar = Pheno(self.cov_fn).read()
        covar = SnpData(iid=covar.iid,sid=["pheno-1"],val=covar.val)
        covar_by_chrom = {chrom:self.cov_fn for chrom in range(1,6)}
        output_file = self.file_name("covar_by_chrom_mixing")
        frame = single_snp(test_snps, pheno,
                                    covar=covar,
                                    covar_by_chrom=covar_by_chrom,
                                    output_file_name=output_file,count_A1=False
                                    )
        self.compare_files(frame,"covar_by_chrom_mixing")


    def compare_files(self,frame,ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp/"+ref_base+".txt")

        #sid_list,pvalue_list = frame['SNP'].values,frame['Pvalue'].values

        #sid_to_pvalue = {}
        #for index, sid in enumerate(sid_list):
        #    sid_to_pvalue[sid] = pvalue_list[index]

        reference=pd.read_csv(reffile,delimiter='\s',comment=None,engine='python')
        assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(reffile)
        frame.set_index('SNP',inplace=True)
        reference.set_index('SNP',inplace=True)
        diff = (frame.PValue-reference.PValue)
        bad = diff[np.abs(diff)>1e-5]
        if len(bad) > 0:
            raise Exception("snps differ too much from file '{0}' at these snps {1}".format(reffile,bad))


def getTestSuite():
    

    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnp)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpLeaveOutOneChrom)
    return unittest.TestSuite([suite1,suite2])



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalInParts


    # this import is needed for the runner
    from fastlmm.association.tests.test_single_snp import TestSingleSnp,TestSingleSnpLeaveOutOneChrom
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=False)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    else: #Cluster test run
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest

        remote_python_parent=r"\\GCR\Scratch\RR1\escience\carlk\data\carlk\pythonpath06292016"
        runner = HPC(2, 'GCR',r"\\GCR\Scratch\RR1\escience",
                                                    remote_python_parent=remote_python_parent,
                                                    unit='node', #core, socket, node
                                                    update_remote_python_parent=True,
                                                    template="Preemptable",
                                                    priority="Lowest",
                                                    nodegroups="Preemptable",
                                                    #excluded_nodes=['gcrcn0231'],
                                                    runtime="0:11:0", # day:hour:min
                                                    max = 10
                                                    )
        #runner = Local()
        #runner = LocalMultiProc(taskcount=2,mkl_num_threads=5,just_one_process=False)
        #runner = LocalInParts(0,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print(runner.run(distributable_test))


    logging.info("done with testing")
