import matplotlib
matplotlib.use("TKAgg",warn=False)
import pylab
import pandas as pd

import numpy as np
import logging
import unittest
import doctest
import os.path
from pysnptools.snpreader import Bed, Pheno,SnpData
from fastlmm.association import single_snp_all_plus_select,single_snp
from fastlmm.feature_selection.test import TestFeatureSelection

from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalInParts, LocalMultiThread
def mf_to_runner_function(mf):
    excluded_nodes=[]#'GCRCM07B20','GCRCM11B05','GCRCM10B06','GCRCM02B07']#'GCRCM02B11','GCRCM03B07'] #'GCRCM22B06','GCRCN0383','GCRCM02B07','GCRCN0179','GCRCM37B13','GCRCN0376','GCRCN0456']#'gcrcn0231']#"MSR-HDP-DN0316","MSR-HDP-DN0321","MSR-HDP-DN0336","MSR-HDP-DN0377","MSR-HDP-DN0378","MSR-HDP-DN0314","MSR-HDP-DN0335","MSRQC073","MSRQC002","MSRQC015"]
    remote_python_parent=r"\\GCR\Scratch\RR1\escience\carlk\data\carlk\pythonpath10262016"
    clean_up=False

    if mf == "debug":
        runner_function = lambda ignore: LocalInParts(215,215,mkl_num_threads=20,result_file="result.p",run_dir=r"C:\deldir\test\outputx") 
    elif mf == "local":
        runner_function = lambda ignore: Local()
    elif mf == "local1":
        runner_function = lambda ignore: Local(1)
    elif mf == "lmp":
        runner_function = lambda ignore: LocalMultiProc(22,5)
    elif mf == "lmt":
        runner_function = lambda ignore: LocalMultiThread(22,5)
    elif mf == "lmtl":
        runner_function = lambda ignore: LocalMultiThread(22,5,just_one_process=True)
    elif mf == "lmp4":
        runner_function = lambda ignore: LocalMultiProc(4,5)
    elif mf == "lmpl":
        runner_function = lambda taskcount: LocalMultiProc(taskcount,taskcount,just_one_process=True)
    elif mf == "nodeP":
        runner_function = lambda taskcount: HPC(min(taskcount,30100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='node', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Preemptable",
                                            priority="Lowest",
                                            excluded_nodes=excluded_nodes,
                                            #mkl_num_threads=20,
                                            nodegroups="Preemptable",
                                            runtime="0:11:0", # day:hour:min
                                            #min = 10 #max(1,min(taskcount,110)//20)
                                            #max = min(taskcount,500),
                                            clean_up=clean_up,
                                            )
    elif mf == "nodeP99":
        runner_function = lambda taskcount: HPC(min(taskcount,30100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='node', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Preemptable",
                                            priority="Lowest",
                                            excluded_nodes=excluded_nodes,
                                            #mkl_num_threads=20,
                                            nodegroups="Preemptable,B99",
                                            runtime="0:11:0", # day:hour:min
                                            #min = 10 #max(1,min(taskcount,110)//20)
                                            #max = min(taskcount,500),
                                            clean_up=clean_up,
                                            )
    elif mf == "nodeL99":
        runner_function = lambda taskcount: HPC(min(taskcount,30100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='node', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="LongRunQ",
                                            priority="Lowest",
                                            excluded_nodes=excluded_nodes,
                                            #mkl_num_threads=20,
                                            nodegroups="LongRunQ,B99",
                                            runtime="11:0:0", # day:hour:min
                                            #min = 10 #max(1,min(taskcount,110)//20)
                                            #max = min(taskcount,500),
                                            clean_up=clean_up,
                                            )
    elif mf == "socketP":
        runner_function = lambda taskcount: HPC(min(taskcount,30100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='socket', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Preemptable",
                                            priority="Lowest",
                                            excluded_nodes=excluded_nodes,
                                            mkl_num_threads=10,
                                            nodegroups="Preemptable",
                                            runtime="0:11:0", # day:hour:min
                                            #min = max(1,min(taskcount,110)//20),
                                            clean_up=clean_up,
                                            )
    elif mf == "coreP":
        runner_function = lambda taskcount: HPC(min(taskcount,1000), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='core', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Preemptable",
                                            priority="Lowest",
                                            excluded_nodes=excluded_nodes,
                                            mkl_num_threads=1,
                                            runtime="0:11:0", # day:hour:min
                                            nodegroups="Preemptable",
                                            #min = min(taskcount,1100)
                                            min = 1,
                                            max = 200 * 20,
                                            clean_up=clean_up,
                                            )
    elif mf == "coreP99":
        runner_function = lambda taskcount: HPC(min(taskcount,1000), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='core', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Preemptable",
                                            priority="Lowest",
                                            excluded_nodes=excluded_nodes,
                                            mkl_num_threads=1,
                                            runtime="0:11:0", # day:hour:min
                                            nodegroups="Preemptable,B99",
                                            #min = min(taskcount,1100)
                                            min = 1,
                                            max = 200 * 20,
                                            clean_up=clean_up,
                                            )
    elif mf == "coreAz":
        runner_function = lambda taskcount: HPC(min(taskcount,1000), 'GCR',r"\\GCR\Scratch\AZ-USCentral\escience",
                                            remote_python_parent=r"\\GCR\Scratch\AZ-USCentral\escience\carlk\data\carlk\pythonpath",
                                            unit='core', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Azure IaaS USCentral",
                                            mkl_num_threads=1,
                                            runtime="0:8:0", # day:hour:min,
                                            clean_up=clean_up,
                                            )
    elif mf == "nodeE":
        runner_function = lambda taskcount: HPC(min(taskcount,10100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='node', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="ExpressQ",
                                            priority="Normal",
                                            #node_local = False,
                                            #mkl_num_threads=20,
                                            runtime="0:4:0", # day:hour:min
                                            #min = min(taskcount,100),
                                            clean_up=clean_up,
                                            )
    elif mf == "50tasks":
        runner_function = lambda taskcount: HPC(50, 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='node', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="ExpressQ",
                                            priority="Normal",
                                            #mkl_num_threads=20,
                                            runtime="0:4:0", # day:hour:min
                                            #min = min(taskcount,100),
                                            clean_up=clean_up,
                                            )
    elif mf == "coreE":
        runner_function = lambda taskcount: HPC(min(taskcount,10100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='core', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="ExpressQ",
                                            priority="Normal",
                                            mkl_num_threads=1,
                                            runtime="0:4:0", # day:hour:min
                                            #min = min(taskcount,100),
                                            clean_up=clean_up,
                                            )
    elif mf == "nodeA":
        runner_function = lambda taskcount: HPC(min(taskcount,30100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='node', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Admin Template",
                                            clean_up=clean_up,
                                            )
    elif mf == "socketA":
        runner_function = lambda taskcount: HPC(min(taskcount,30100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='socket', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Admin Template",
                                            clean_up=clean_up,
                                            )
    elif mf == "coreA":
        runner_function = lambda taskcount: HPC(min(taskcount,30100), 'GCR',r"\\GCR\Scratch\RR1\escience",
                                            remote_python_parent=remote_python_parent,
                                            unit='core', #core, socket, node
                                            update_remote_python_parent=True,
                                            template="Admin Template",
                                            clean_up=clean_up,
                                            )
    elif mf == "nodeH":
        runner_function = lambda taskcount: Hadoop2(min(taskcount,100000), mapmemory=58*1024, reducememory=8*1024, min_alloc=2048, xmx=3072, mkl_num_threads=14, queue="shared",skipdatacheck=True,skipsourcecheck=True)
    elif mf == "coreH":
        runner_function = lambda taskcount: Hadoop2(min(taskcount,100000), mapmemory=8*1024, reducememory=8*1024, min_alloc=2048, xmx=3072, mkl_num_threads=1, queue="shared",skipdatacheck=True,skipsourcecheck=True)
    else:
        raise Exception("don't find mf="+mf)
    return runner_function

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

    def test_notebook(self):
        do_plot = False
        mf_name = "lmp" #"local", "coreP", "nodeP", "socketP", "nodeE", "lmp"
        runner = mf_to_runner_function(mf_name)(4)
        output_file_name = self.file_name("notebook")


        logging.info("TestSingleSnpAllPlusSelect test_one")
        # define file names
        snp_reader = Bed(self.pythonpath + "/tests/datasets/synth/all",count_A1=False)
        pheno_fn = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"
        cov_fn = self.pythonpath + "/tests/datasets/synth/cov.txt"

        # find the chr5 SNPs
        test_snps = snp_reader[:,snp_reader.pos[:,0] == 5]

        #select the 2nd kernel and run GWAS
        results = single_snp_all_plus_select(test_snps=test_snps,G=snp_reader,pheno=pheno_fn,GB_goal=2,do_plot=do_plot,output_file_name=output_file_name,runner=runner, count_A1=False)


        self.compare_files(results,"notebook")

    def test_one(self):
        from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc

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

    def test_three(self): #!!! rather a big test case
        from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
        logging.info("TestSingleSnpAllPlusSelect test_three")

        bed_fn = self.pythonpath + "/tests/datasets/synth/all.bed"
        bed_fn = Bed(bed_fn,count_A1=False)
        pheno_fn = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"
        cov_fn = self.pythonpath + "/tests/datasets/synth/cov.txt"

        mf_name = "lmp" #"local", "coreP", "nodeP", "socketP", "nodeE", "lmp"
        runner = mf_to_runner_function(mf_name)(4)


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

    def test_two(self): #!!! rather a big test case
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

        mf_name = "lmpl" #"lmpl" "local", "coreP", "nodeP", "socketP", "nodeE", "lmp"
        runner = mf_to_runner_function(mf_name)(20)

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

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testfile("../single_snp_all_plus_select.py")
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__



def getTestSuite():
    
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpAllPlusSelect)
    return unittest.TestSuite([suite1])



if __name__ == '__main__':

    # this import is needed for the runner
    from fastlmm.association.tests.test_single_snp_all_plus_select import TestSingleSnpAllPlusSelect
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=True)
        r.run(suites)
    else: #Cluster test run



        from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest


        runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        distributable_test = DistributableTest(suites,"temp_test")
        print runner.run(distributable_test)


    logging.info("done with testing")
