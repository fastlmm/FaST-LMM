from __future__ import absolute_import
import numpy as np
import scipy as sp
import logging
from scipy import stats
from fastlmm.pyplink.snpreader.Bed import Bed
#from fastlmm.association.gwas import LeaveOneChromosomeOut, LocoGwas, FastGwas, load_intersect
from fastlmm.association.LeaveOneChromosomeOut import LeaveOneChromosomeOut
from fastlmm.association.PrecomputeLocoPcs import load_intersect
from fastlmm.association.LocoGwas import FastGwas, LocoGwas
from fastlmm.util import run_fastlmmc
from fastlmm.inference import LMM
import unittest
import os.path
import time
from six.moves import range
import platform

currentFolder = os.path.dirname(os.path.realpath(__file__))


class TestGwas(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        #self.snpreader_bed = Bed(currentFolder + "../feature_selection/examples/toydata.5chrom.bed")
        #self.pheno_fn = currentFolder + "../feature_selection/examples/toydata.phe"
 
        self.meh = True

    def test_loco(self):
        """
        test leave one chromosome out iterator
        """

        names = ["a", "b", "a", "c", "b", "c", "b"]

        loco = LeaveOneChromosomeOut(names)

        expect = [[[1,3,4,5,6],[0,2]], 
                  [[0,2,3,5],[1,4,6]],
                  [[0,1,2,4,6],[3,5]]]

        for i, (train_idx, test_idx) in enumerate(loco):
            assert (expect[i][0] == train_idx).all()
            assert (expect[i][1] == test_idx).all()
    

    #def xtest_results_identical_with_fastlmmcX(self):
    #    """
    #    make sure gwas yields same results as fastlmmC
    #    """

    #    os.chdir(r"d:\data\carlk\cachebio\genetics\wtccc\data")
    #    bed_fn = "filtered/wtcfb"
    #    pheno_fn = r'pheno\cad.txt'

    #    logging.info("Loading Bed")
    #    snp_reader = Bed(bed_fn)
    #    import fastlmm.pyplink.snpset.PositionRange as PositionRange
    #    snp_set = PositionRange(0,201)
    #    logging.info("Intersecting and standardizing")
    #    G, y, _, _ = load_intersect(snp_reader, pheno_fn, snp_set)

    #    snp_pos = snp_reader.rs

    #    idx_sim = range(0, 200)
    #    idx_test = range(200,201)

    #    #snp_pos_sim = snp_pos[idx_sim]
    #    #snp_pos_test = snp_pos[idx_test]

    #    G_chr1, G_chr2 = G[:,idx_sim], G[:,idx_test]
    #    delta = 4.0

    #    REML = False
    #    #gwas_c = GwasTest(bed_fn, pheno_fn, snp_pos_sim, snp_pos_test, delta, REML=REML)
    #    #gwas_c.run_gwas()

    #    logging.info("Creating GwasPrototype")
    #    gwas = GwasPrototype(G_chr1, G_chr2, y, delta, REML=REML)
    #    logging.info("running GwasPrototype")
    #    gwas.run_gwas()
    #    logging.info("finished GwasPrototype")

    #    #gwas_f = FastGwas(G_chr1, G_chr2, y, delta, findh2=False)
    #    #gwas_f.run_gwas()

    #    sorted_snps = snp_pos_test[gwas.p_idx]

        
        ## make sure we get p-values right
        #np.testing.assert_array_almost_equal(gwas.p_values, gwas_c.p_values, decimal=3)
        #np.testing.assert_array_almost_equal(gwas.p_values, gwas_f.p_values, decimal=3)

        #np.testing.assert_array_almost_equal(gwas.sorted_p_values, gwas_c.sorted_p_values, decimal=3)
        #np.testing.assert_array_almost_equal(gwas.sorted_p_values, gwas_f.sorted_p_values, decimal=3)


    def test_results_identical_with_fastlmmc(self):
        """
        make sure gwas yields same results as fastlmmC
        """
        if platform.system() == "Darwin": #Don't run old C code on Mac
            return

        currentFolder = os.path.dirname(os.path.realpath(__file__))

        #prefix = r"C:\Users\chwidmer\Documents\Projects\sandbox\data\test"
        #bed_fn = prefix + "/jax_gt.up.filt.M"
        #dat_fn = prefix + "/jax_M_expression.1-18.dat"
        #pheno_fn = prefix + "/jax_M_expression.19.phe.txt"
        
        bed_fn = os.path.join(currentFolder, "../../feature_selection/examples/toydata.5chrom")
        pheno_fn = os.path.join(currentFolder, "../../feature_selection/examples/toydata.phe")

        #prefix = "../../../tests\datasets\mouse"
        #bed_fn = os.path.join(prefix, "alldata")
        #pheno_fn = os.path.join(prefix, "pheno.txt")

        snp_reader = Bed(bed_fn)
        G, y, _, _ = load_intersect(snp_reader, pheno_fn)

        snp_pos = snp_reader.rs

        
        idx_sim = list(range(0, 5000))
        idx_test = list(range(5000, 10000))

        snp_pos_sim = snp_pos[idx_sim]
        snp_pos_test = snp_pos[idx_test]

        G_chr1, G_chr2 = G[:,idx_sim], G[:,idx_test]
        delta = 1.0



        ###################################
        # REML IN lmm.py is BROKEN!!

        # we compare REML=False in lmm.py to fastlmmc
        REML = False
        gwas_c_reml = GwasTest(bed_fn, pheno_fn, snp_pos_sim, snp_pos_test, delta, REML=REML)
        gwas_c_reml.run_gwas()

        gwas = GwasPrototype(G_chr1, G_chr2, y, delta, REML=False)
        gwas.run_gwas()

        # check p-values in log-space!
        np.testing.assert_array_almost_equal(np.log(gwas.p_values), np.log(gwas_c_reml.p_values), decimal=3)
        if False:
            import pylab
            pylab.plot(np.log(gwas_c_reml.p_values), np.log(gwas_f.p_values_F), "x")
            pylab.plot(list(range(-66,0,1)), list(range(-66,0,1)))
            pylab.show()

        # we compare lmm_cov.py to fastlmmc with REML=False
        gwas_c = GwasTest(bed_fn, pheno_fn, snp_pos_sim, snp_pos_test, delta, REML=True)
        gwas_c.run_gwas()
        gwas_f = FastGwas(G_chr1, G_chr2, y, delta, findh2=False)
        gwas_f.run_gwas()
        np.testing.assert_array_almost_equal(np.log(gwas_c.p_values), np.log(gwas_f.p_values_F), decimal=2)

        # additional testing code for the new wrapper functions

        # Fix delta
        from pysnptools.snpreader import Bed as BedSnpReader
        from fastlmm.association.single_snp import single_snp
        snpreader = BedSnpReader(bed_fn,count_A1=False)
        frame = single_snp(test_snps=snpreader[:,idx_test], pheno=pheno_fn, G0=snpreader[:,idx_sim],h2=1.0/(delta+1.0),leave_out_one_chrom=False,count_A1=False)
        sid_list,pvalue_list = frame['SNP'].values,frame['PValue'].values
        np.testing.assert_allclose(gwas_f.sorted_p_values_F, pvalue_list, rtol=1e-10)

        p_vals_by_genomic_pos = frame.sort_values(["Chr", "ChrPos"])["PValue"].tolist()
        np.testing.assert_allclose(gwas_c_reml.p_values, p_vals_by_genomic_pos, rtol=.1)
        np.testing.assert_allclose(gwas_c_reml.p_values, gwas_f.p_values_F, rtol=.1)
        np.testing.assert_allclose(gwas_f.sorted_p_values_F, gwas_c_reml.sorted_p_values, rtol=.1)


        # Search over delta
        gwas_c_reml_search = GwasTest(bed_fn, pheno_fn, snp_pos_sim, snp_pos_test, delta=None, REML=True)
        gwas_c_reml_search.run_gwas()

        frame_search = single_snp(test_snps=snpreader[:,idx_test], pheno=pheno_fn, G0=snpreader[:,idx_sim],h2=None,leave_out_one_chrom=False,count_A1=False)
        _,pvalue_list_search = frame_search['SNP'].values,frame_search['PValue'].values

        p_vals_by_genomic_pos = frame_search.sort_values(["Chr", "ChrPos"])["PValue"].tolist()
        np.testing.assert_allclose(gwas_c_reml_search.p_values, p_vals_by_genomic_pos, rtol=.001)
        np.testing.assert_allclose(gwas_c_reml_search.sorted_p_values, pvalue_list_search, rtol=.001)


class GwasPrototype(object):
    """
    class to perform genome-wide scan
    """
    

    def __init__(self, train_snps, test_snps, phen, delta=None, cov=None, REML=False, train_pcs=None, mixing=0.0):
        """
        set up GWAS object
        """

        self.REML = REML
        self.train_snps = train_snps
        self.test_snps = test_snps
        self.phen = phen
        if delta is None:
            self.delta=None
        else:
            self.delta = delta * train_snps.shape[1]
        self.n_test = test_snps.shape[1]
        self.n_ind = len(self.phen)

        self.train_pcs = train_pcs
        self.mixing = mixing

        # add bias if no covariates are used
        if cov is None:
            self.cov = np.ones((self.n_ind, 1))
        else:
            self.cov = cov
        self.n_cov = self.cov.shape[1] 
       
        self.lmm = None
        self.res_null = None
        self.res_alt = []

        self.ll_null = None
        self.ll_alt = np.zeros(self.n_test)
        self.p_values = np.zeros(self.n_test)
        self.sorted_p_values = np.zeros(self.n_test)

        # merge covariates and test snps
        self.X = np.hstack((self.cov, self.test_snps))

    
    def precompute_UX(self, X): 
        ''' 
        precompute UX for all snps to be tested
        --------------------------------------------------------------------------
        Input:
        X       : [N*D] 2-dimensional array of covariates
        --------------------------------------------------------------------------
        '''

        logging.info("precomputing UX")

        self.UX = self.lmm.U.T.dot(X)
        self.k = self.lmm.S.shape[0]
        self.N = self.lmm.X.shape[0]
        if (self.k<self.N):
            self.UUX = X - self.lmm.U.dot(self.UX)

        logging.info("done.")


    def train_null(self):
        """
        train model under null hypothesis
        """

        logging.info("training null model")

        # use LMM
        self.lmm = LMM()
        self.lmm.setG(self.train_snps, self.train_pcs, a2=self.mixing)

        self.lmm.setX(self.cov)
        self.lmm.sety(self.phen)

        logging.info("finding delta")
        if self.delta is None:
            result = self.lmm.findH2(REML=self.REML, minH2=0.00001 )
            self.delta = 1.0/result['h2']-1.0
            
        # UX = lmm_null.U.dot(test_snps)
        self.res_null = self.lmm.nLLeval(delta=self.delta, REML=self.REML)
        self.ll_null = -self.res_null["nLL"]


    def set_current_UX(self, idx):
        """
        set the current UX to pre-trained LMM
        """

        si = idx + self.n_cov

        self.lmm.X = np.hstack((self.X[:,0:self.n_cov], self.X[:,si:si+1]))
        self.lmm.UX = np.hstack((self.UX[:,0:self.n_cov], self.UX[:,si:si+1]))
        if (self.k<self.N):
            self.lmm.UUX = np.hstack((self.UUX[:,0:self.n_cov], self.UUX[:,si:si+1]))
    

    def train_alt(self):
        """
        train alternative model
        """ 
   
        assert self.lmm != None
        self.precompute_UX(self.X)

        for idx in range(self.n_test):

            self.set_current_UX(idx)
            res = self.lmm.nLLeval(delta=self.delta, REML=self.REML)

            self.res_alt.append(res)
            self.ll_alt[idx] = -res["nLL"]

            if idx % 1000 == 0:
                logging.info("processing snp {0}".format(idx))


    def compute_p_values(self):
        """
        given trained null and alt models, compute p-values
        """

        # from C++ (?)
        #real df = rank_beta[ snp ] - ((real)1.0 * rank_beta_0[ snp ]) ;
        #pvals[ snp ] = PvalFromLikelihoodRatioTest( LL[ snp ] - LL_0[ snp ], ((real)0.5 * df) );

        degrees_of_freedom = 1

        assert len(self.res_alt) == self.n_test

        for idx in range(self.n_test):
            test_statistic = self.ll_alt[idx] - self.ll_null
            self.p_values[idx] = stats.chi2.sf(2.0 * test_statistic, degrees_of_freedom)

        
        self.p_idx = np.argsort(self.p_values)
        self.sorted_p_values = self.p_values[self.p_idx]
        


    def plot_result(self):
        """
        plot results
        """
        
        import pylab
        pylab.semilogy(self.p_values)
        pylab.show()

        dummy = [self.res_alt[idx]["nLL"] for idx in range(self.n_test)]
        pylab.hist(dummy, bins=100)
        pylab.title("neg likelihood")
        pylab.show()

        pylab.hist(self.p_values, bins=100)
        pylab.title("p-values")
        pylab.show()
 

    def run_gwas(self):
        """
        invoke all steps in the right order
        """

        self.train_null()
        self.train_alt()
        self.compute_p_values()
        #self.plot_result()



class GwasTest(object):
    """
    genome-wide scan using FastLmmC

    --------------------------------------------------------------------------
    Input:
    bfile       basename for PLINK's binary .bed,.fam and .bin files
    pheno       name of phenotype file
    bfileSim    basename for PLINK's binary files for building genetic similarity
    sim         specifies that genetic similarities are to be read directly from this file
    linreg      specifies that linear regression will be performed. when this option is used, no genetic similarities should be specified (boolean)
    covar       optional file containing the covariates
    out         the name of the output file
    optLogdelta if set, delta is not optimized
    extract     FaSTLMM will only analyze the SNPs explicitly listed in the filename
    extractSim  FastLMM will only use the SNPs explicitly listed for computing genetic similarity
    autoSelect  determines the SNPs to be included in the similarity matrix. When this option is used, GWAS is not run. SNPs are written to filename.snps.txt
                and statistics to filename.xval.txt
    autoSelectCriterionMSE  directs AutoSelect to use out-of-sample mean-squared error for the selection criterion. Otherwise out-of-sample log likelihood is used
    excludeByPosition       excludes the SNP tested and those within this distance from the genetic similarity matrix
    excludeByGeneticDistance excludes the SNP tested and those within this distance from the genetic similarity matrix
    eigen       load the spectral decomposition object from the directory name
    eigenOut    save the spectral decomposition object to the directory name
    maxThreads  suggests the level of parallelism to use
    simOut      specifies that genetic similarities are to be written to this file
    topKbyLinRegdirects AutoSelect to use only the top <int> SNPs, as determined by linear regression, while selecting SNPs
    for more information, we refer to the user-manual of fast-lmm

    # run linear mixed model
    run_fastlmmc.run(bfile=self.bedFile,pheno=self.phenoFile,bfileSim=self.bedFileSim,linreg=self.linreg,
                    covar=self.covarFile,optLogdelta=self.logdelta,extractSim=self.extractSim,
                    excludeByPosition=excludeByPosition,excludeByGeneticDistance=excludeByGeneticDistance,
                    fastlmm_path=self.fastlmmPath,out=self.outFile,extract=self.extract,numJobs=self.numJobs,
                    thisJob=self.thisJob)
    """


    def __init__(self, bed_fn, pheno_fn, snp_idx_sim, snp_idx_test, delta, REML=False, excludeByPosition=None):
        "make a call to fastlmm c"


        self.extract = "tmp_extract.txt"
        self.extractSim = "tmp_extract_sim.txt"

        self.write_snp_file(self.extract, snp_idx_test)
        self.write_snp_file(self.extractSim, snp_idx_sim)

        self.bedFile = bed_fn
        self.bedFileSim = bed_fn
        self.phenoFile = pheno_fn
        self.optLogdelta = np.log(delta) if delta is not None else None
        self.REML =REML

        currentFolder = os.path.dirname(os.path.realpath(__file__))
        self.fastlmm_path = os.path.join(currentFolder,"../Fastlmm_autoselect")
        self.out_file = "out.txt"

        self.sorted_p_values = None
        self.excludeByPosition = excludeByPosition
    

    def write_snp_file(self, out_fn, snp_ids):
        """
        write out snps to flat file 
        """

        with open(out_fn, "w") as f:
            for sid in snp_ids:
                f.write(str(sid) + "\n")


    def run_gwas(self):
        """
        """

        # run linear mixed model
        run_fastlmmc.run(bfile=self.bedFile, pheno=self.phenoFile, bfileSim=self.bedFileSim,
                        optLogdelta=self.optLogdelta, extractSim=self.extractSim,
                        fastlmm_path=self.fastlmm_path, out=self.out_file, extract=self.extract, 
                        REML=self.REML, excludeByPosition=self.excludeByPosition)

        self.read_results()


    def read_results(self):
        """
        read results file
        """

        import pandas as pd
        
        table = pd.read_csv(self.out_file,sep='\t')
        self.sorted_p_values = table["Pvalue"].tolist()
        self.sorted_snps = table["SNP"].tolist()
        
        self.p_values = table.sort_values(["Chromosome", "Position"])["Pvalue"].tolist() 


def getTestSuite():
    """
    set up composite test suite
    """
    
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestGwas)
    return unittest.TestSuite([suite1])

if __name__ == '__main__':    
    unittest.main()
