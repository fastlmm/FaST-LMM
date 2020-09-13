"""
implementation of windowing to cut out only single snp from G0 (which is used to condition on AND to test)

Authors: Chris Widmer
Created: 9/25/2014
"""

from __future__ import absolute_import
import os
import numpy as np
import logging
from scipy import stats
from pysnptools.snpreader import Bed
from pysnptools.util import intersect_apply
import pysnptools.util.pheno as pstpheno
import fastlmm.util.standardizer as stdizer
from fastlmm.inference import LMM
from six.moves import range


class WindowingGwas(object):
    """
    class to perform genome-wide scan with single-snp windowing
    """
    

    def __init__(self, G0, phen, delta=None, cov=None, REML=False, G1=None, mixing=0.0):
        """
        set up GWAS object
        """

        self.REML = REML
        self.G0 = G0
        self.test_snps = G0
        self.phen = phen
        if delta is None:
            self.delta=None
        else:
            self.delta = delta * G0.shape[1]
        self.n_test = self.test_snps.shape[1]
        self.n_ind = len(self.phen)

        self.G1 = G1
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

        self.ll_null = np.zeros(self.n_test)
        self.ll_alt = np.zeros(self.n_test)
        self.p_values = np.zeros(self.n_test)
        self.sorted_p_values = np.zeros(self.n_test)

        # merge covariates and test snps
        self.X = np.hstack((self.cov, self.test_snps))
        self.N = self.X.shape[0]
    
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
        find delta on all snps
        """

        logging.info("training null model")

        # use LMM
        self.lmm = LMM()
        self.lmm.setG(self.G0, self.G1, a2=self.mixing)

        self.lmm.setX(self.cov)
        self.lmm.sety(self.phen)

        logging.info("finding delta")

        #result = self.lmm.find_log_delta(self, self.N)
        #self.delta = np.exp(result['log_delta'])

        if self.delta is None:
            result = self.lmm.find_log_delta_chris()
            self.delta = result['delta']



    def set_current_UX(self, idx):
        """
        set the current UX to pre-trained LMM
        """

        si = idx + self.n_cov

        self.lmm.X = np.hstack((self.X[:,0:self.n_cov], self.X[:,si:si+1]))
        self.lmm.UX = np.hstack((self.UX[:,0:self.n_cov], self.UX[:,si:si+1]))
        if (self.k<self.N):
            self.lmm.UUX = np.hstack((self.UUX[:,0:self.n_cov], self.UUX[:,si:si+1]))
    

    def set_null_UX(self):
        """
        reset UX to covariates only
        """
        self.lmm.X = self.X[:,0:self.n_cov]
        self.lmm.UX = self.UX[:,0:self.n_cov]
        if (self.k<self.N):
            self.lmm.UUX = self.UUX[:,0:self.n_cov]
    

    def train_windowing(self):
        """
        train null and alternative model
        """ 
   
        assert self.lmm != None
        self.precompute_UX(self.X)

        for idx in range(self.n_test):

            #TODO: this can be generalized to bigger window
            self.lmm.set_exclude_idx([idx])

            # null model
            self.set_null_UX()
            res = self.lmm.nLLeval(delta=self.delta, REML=self.REML)
            self.ll_null[idx] = -res["nLL"]

            # alternative model
            self.set_current_UX(idx)
            res = self.lmm.nLLeval(delta=self.delta, REML=self.REML)

            self.res_alt.append(res)
            self.ll_alt[idx] = -res["nLL"]

            if idx % 1000 == 0:
                logging.warning("processing snp {0}".format(idx))


    def compute_p_values(self):
        """
        given trained null and alt models, compute p-values
        """

        degrees_of_freedom = 1

        assert len(self.res_alt) == self.n_test

        for idx in range(self.n_test):
            test_statistic = self.ll_alt[idx] - self.ll_null[idx]
            self.p_values[idx] = stats.chi2.sf(2.0 * test_statistic, degrees_of_freedom)

        self.p_idx = np.argsort(self.p_values)
        self.sorted_p_values = self.p_values[self.p_idx]
        
        return self.p_values


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
        self.train_windowing()
        return self.compute_p_values()

        
def simple_manhattan_plot(p_values):
    """
    plot p-values
    """

    import pylab
    
    pylab.plot(-np.log(p_values), "bx")

    pylab.show()


def main():
    """
    example that compares output to fastlmmc
    """


    # set up data
    phen_fn = "../feature_selection/examples/toydata.phe"
    snp_fn = "../feature_selection/examples/toydata.5chrom.bed"
    #chrom_count = 5
    
    # load data
    ###################################################################
    snp_reader = Bed(snp_fn)
    pheno = pstpheno.loadOnePhen(phen_fn)

    cov = None
    #cov = pstpheno.loadPhen(self.cov_fn)    

    snp_reader, pheno, cov = intersect_apply([snp_reader, pheno, cov])
    
    G = snp_reader.read(order='C').val
    G = stdizer.Unit().standardize(G)
    G.flags.writeable = False
    y = pheno['vals'][:,0]
    y.flags.writeable

    # load pcs
    #G_pc = cov['vals']
    #G_pc.flags.writeable = False
    delta = 2.0
    gwas = WindowingGwas(G, y, delta=delta)
    pv = gwas.run_gwas()

    from fastlmm.association.tests.test_gwas import GwasTest
    REML = False
    snp_pos_sim = snp_reader.sid
    snp_pos_test = snp_reader.sid
    os.environ["FastLmmUseAnyMklLib"] = "1"
    gwas_c = GwasTest(snp_fn, phen_fn, snp_pos_sim, snp_pos_test, delta, REML=REML, excludeByPosition=0)
    gwas_c.run_gwas()

    import pylab
    pylab.plot(np.log(pv), np.log(gwas_c.p_values), "+")
    pylab.plot(np.arange(-18, 0), np.arange(-18,0), "-k")
    pylab.show()

    np.testing.assert_array_almost_equal(np.log(pv), np.log(gwas_c.p_values), decimal=3)
    
    simple_manhattan_plot(pv)


if __name__ == "__main__":
    main()
