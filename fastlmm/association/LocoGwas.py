#!/usr/bin/env python2.7
#
# Copyright (C) 2014 Microsoft Research

"""
Created on 2014-03-11
@summary: Module for performing GWAS
"""

from __future__ import absolute_import
import time
import logging
import os.path

import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats



import fastlmm.inference as fastlmm

import fastlmm.util.util as util 
from fastlmm.util.pickle_io import load, save
from fastlmm.util.util import argintersect_left

import fastlmm.pyplink.plink as plink
import fastlmm.util.standardizer as stdizer

from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import f_regression
from fastlmm.association.PrecomputeLocoPcs import PrecomputeLocoPcs, load_intersect
from fastlmm.association.LeaveOneChromosomeOut import LeaveOneChromosomeOut
from six.moves import range


class LocoGwas(object): #implements IDistributable
    """
    module to perform GWAS using Leave One Chromosome Out (LOCO) p-value
    evaluation, while keeping track of the coordinates
    """


    def __init__(self,  chrom_count, snp_reader, pheno_fn, selected_snps=None, delta=1.0, num_pcs=0, mixing=0.0, pc_prefix=None, pcs_only=None):

        self.chrom_count = chrom_count

        self.debug = False
        self.use_fast_but_slightly_inaccurate_pca = True
        self.pc_prefix = pc_prefix
        self.pcs_only = pcs_only # if this is True, no snps are used in GWAS

        self.delta = delta
        self.num_pcs = num_pcs
        self.mixing = mixing
        self.selected_snps = selected_snps

        self.snp_reader = snp_reader
        snp_reader.run_once()

        self.pheno_fn = pheno_fn

        # results

        #self.p_values = None
        #self.p_values_lin = None
        #self.rs = None
        #self.pos = None

#start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        return self.chrom_count

    def work_sequence(self):

        # is it OK to do the intersect and the linear regression 23 extra times?


        # clear
        G, y, snp_name, _ = load_intersect(self.snp_reader, self.pheno_fn)

        # compute linear regression
        _, p_values_lin = f_regression(G, y, center=True)

        # set up empty return structures
        #self.rs = snp_name
        #self.p_values = -np.ones(len(snp_name))

        # get chr names/id
        chr_ids = self.snp_reader.pos[:,0]

        #self.pos = self.snp_reader.pos

        #loco = [[range(0,5000), range(5000,10000)]]
        loco = LeaveOneChromosomeOut(chr_ids, indices=True)

        if len(loco) is not self.chrom_count :  raise Exception("The snp reader has {0} chromosome, not {1} as specified".format(len(loco),self.chrom_count))

    
        for i, (train_snp_idx, test_snp_idx) in enumerate(loco):
            if i == 0:
                result = {"p_values":-np.ones(len(snp_name)),
                          "p_values_lin": p_values_lin,
                          "rs":snp_name,
                          "pos":self.snp_reader.pos}
            else:
                result = None
            yield lambda i=i, train_snp_idx=train_snp_idx,test_snp_idx=test_snp_idx,result=result,G=G,y=y: self.dowork(i,train_snp_idx,test_snp_idx,result,G,y)  # the 'i=i',etc is need to get around a strangeness in Python

    def reduce(self, result_sequence):
        '''
        '''

        result = None
        for test_snp_idx, p_values,partial_result in result_sequence:
            if partial_result is not None:
                assert(result is None)
                result = partial_result
            result["p_values"][test_snp_idx] = p_values

        # make sure there are no original fields left
        assert (result["p_values"] != -1).all()

        return result

    #optional override -- the str name of the instance is used by the cluster as the job name
    def __str__(self):
        return "{0}({1},'{2}','{3},...')".format(self.__class__.__name__, self.chrom_count, self.snp_reader, self.pheno_fn)
 #end of IDistributable interface---------------------------------------

    def dowork(self, i, train_snp_idx, test_snp_idx, result, G, y):
        logging.info("{0}, {1}".format(len(train_snp_idx), len(test_snp_idx)))
        
        # intersect selected SNPs with train snps
        if not self.selected_snps is None:
            # intersect snp names
            logging.info("intersecting train snps with selected snps for LOCO")
            int_snp_idx = argintersect_left(self.snp_reader.rs[train_snp_idx], self.selected_snps)
            sim_keeper_idx = np.array(train_snp_idx)[int_snp_idx]

        else:
            sim_keeper_idx = train_snp_idx

        # subset data
            
        # fast indexing (needs to be C-order)
        assert np.isfortran(G) == False
        #G_train = G.take(train_snp_idx, axis=1)
        G_sim = G.take(sim_keeper_idx, axis=1)
        G_test = G.take(test_snp_idx, axis=1)

        t0 = time.time()

        if self.num_pcs == 0:
            pcs = None
        else:
            if not self.pc_prefix is None:
                out_fn = PrecomputeLocoPcs.create_out_fn(self.pc_prefix, i)
                logging.info("loading pc from file: %s" % out_fn)
                pcs = load(out_fn)[:,0:self.num_pcs]
                logging.info("..done")

            else:
                assert False, "please precompute PCs"

                logging.info("done after %.4f seconds" % (time.time() - t0))

        # only use PCs
        if self.pcs_only:
            G_sim = None
            logging.info("Using PCs only in LocoGWAS")
        gwas = FastGwas(G_sim, G_test, y, self.delta, train_pcs=pcs, mixing=self.mixing)
        gwas.run_gwas()

        assert len(gwas.p_values) == len(test_snp_idx)

        # wrap up results
        return test_snp_idx, gwas.p_values, result

    
    #def run_loco_gwas(self):

    #    # clear
    #    G, y, snp_name, _ = load_intersect(self.snp_reader, self.pheno_fn)

    #    # compute linear regression
    #    _, self.p_values_lin = f_regression(G, y, center=True)

    #    # set up empty return structures
    #    self.rs = snp_name
    #    self.p_values = -np.ones(len(snp_name))

    #    # get chr names/id
    #    chr_ids = self.snp_reader.pos[:,0]

    #    self.pos = self.snp_reader.pos

    #    #loco = [[range(0,5000), range(5000,10000)]]
    #    loco = LeaveOneChromosomeOut(chr_ids, indices=True)

    #    for i, (train_snp_idx, test_snp_idx) in enumerate(loco):

    #        print len(train_snp_idx), len(test_snp_idx)
        
    #        # intersect selected SNPs with train snps
    #        if not self.selected_snps is None:
    #            # intersect snp names
    #            int_snp_idx = argintersect_left(self.rs[train_snp_idx], self.selected_snps)
    #            sim_keeper_idx = np.array(train_snp_idx)[int_snp_idx]
    #        else:
    #            sim_keeper_idx = train_snp_idx

    #        # subset data
            
    #        # fast indexing (needs to be C-order)
    #        assert np.isfortran(G) == False
    #        G_train = G.take(train_snp_idx, axis=1)
    #        G_sim = G.take(sim_keeper_idx, axis=1)
    #        G_test = G.take(test_snp_idx, axis=1)

    #        t0 = time.time()

    #        if self.num_pcs == 0:
    #            pcs = None
    #        else:
    #            if not self.pc_prefix is None:
    #                out_fn = "%s_%04d.pickle.bzip" % (self.pc_prefix, i)
    #                logging.info("loading pc from file: %s" % out_fn)
    #                pcs = load(out_fn)[:,0:self.num_pcs]
    #                logging.info("..done")

    #            else:
    #                logging.info("computing pca...")
    #                if self.use_fast_but_slightly_inaccurate_pca:

    #                    logging.info("using ARPACK for pca computation")
    #                    K = np.dot(G_train, G_train.T)
    #                    pca = KernelPCA(n_components=self.num_pcs)
    #                    pca._fit_transform(K)
    #                    pcs = pca.alphas_ * np.sqrt(pca.lambdas_)

    #                    logging.info("num_pcs: %i, mixing: %0.4f" % (self.num_pcs, self.mixing))
    #                    #TODO: init with starting value
    #                    #TODO: save kernel
    #                    #self.eigval = pca.lambdas_

    #                    if self.debug:
    #                        pca_old = PCA(n_components=self.num_pcs)
    #                        pcs_old = pca_old.fit_transform(G_train)
    #                        np.testing.assert_array_almost_equal(np.abs(pcs), np.abs(pcs_old), decimal=3)

    #                else:
    #                    pca = PCA(n_components = self.num_pcs)
    #                    pcs = pca.fit_transform(G_train)

    #                logging.info("done after %.4f seconds" % (time.time() - t0))

    #        gwas = FastGwas(G_sim, G_test, y, self.delta, train_pcs=pcs, mixing=self.mixing)
    #        gwas.run_gwas()

    #        # wrap up results
    #        self.p_values[test_snp_idx] = gwas.p_values

    #    # make sure there are no original fields left
    #    assert (self.p_values != -1).all()

    #    return self.p_values


   
    def eval_gwas(self, causal_idx, out_fn=None, plot=False, mindist=10.0):
        """
        wrapper for function
        """

        res = eval_gwas(self.p_values, self.pos, causal_idx, mindist=mindist, out_fn=out_fn, plot=plot)
        res["delta"] = self.delta
        res["num_pcs"] = self.num_pcs
        res["num_selected_snps"] = self.selected_snps
        res["mixing"] = self.mixing
            
        from fastlmm.util.pickle_io import save
        save(out_fn, res)

        return res

    #!! would be nice of this was optional and if not given the OS was asked
    # required by IDistributable
    @property
    def tempdirectory(self):
        if self.pc_prefix is None:
            return ".work_directory.None"
        else:
            return ".work_directory." + self.pc_prefix


    def copyinputs(self, copier):
        if not self.num_pcs == 0:
            for i in range(self.chrom_count):
                pc_fn = PrecomputeLocoPcs.create_out_fn(self.pc_prefix, i)
                copier.input(pc_fn)
        copier.input(self.pheno_fn)
        copier.input(self.snp_reader)

    #Note that the files created are not automatically copied. Instead,
    # whenever we want another file to be created, a second change must be made here so that it will be copied.
    def copyoutputs(self,copier):
        pass


class FastGwas(object):
    """
    class to perform genome-wide scan
    """
    

    def __init__(self, train_snps, test_snps, phen, delta=None, cov=None, train_pcs=None, mixing=0.0, findh2=False):
        """
        set up GWAS object
        """

        self.findh2 = findh2
        self.train_snps = train_snps
        self.test_snps = test_snps
        self.phen = phen
        if delta is None:
            self.delta=None
            logging.info("finding delta")
            self.findh2 = True
        else:
            self.delta = delta * train_snps.shape[1] #!!should also add in the number of train_pcs

        self.n_test = test_snps.shape[1]
        if self.phen.ndim<2:
            self.phen=self.phen[:,np.newaxis]
        self.n_ind,self.n_phen = self.phen.shape

        self.train_pcs = train_pcs
        self.mixing = mixing

        # add bias if no covariates are used
        if cov is None:
            self.cov = np.ones((self.n_ind, 1))
        else:
            self.cov = cov
        self.n_cov = self.cov.shape[1] 
       

        self.lmm = None
        self.p_values = np.zeros(self.n_test)
        self.sorted_p_values = np.zeros(self.n_test)
        # merge covariates and test snps
        #self.X = np.hstack((self.cov, self.test_snps))


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

        from fastlmm.inference.lmm_cov import LMM as fastLMM

        if self.train_pcs is None and self.train_snps is not None:
            assert self.mixing == 0.0
            G = self.train_snps        
        elif self.train_pcs is not None and self.train_snps is None:
            assert self.mixing == 0.0
            G = self.train_pcs
        else:
            logging.info("concat pcs, mixing {0}".format(self.mixing))
            G = np.concatenate((np.sqrt(1.0-self.mixing) * self.train_snps, np.sqrt(self.mixing) * self.train_pcs),1)

        #TODO: make sure low-rank case is handled correctly
        lmm = fastLMM(X=self.cov, Y=self.phen, G=G, K=None)

        if self.findh2:
            opt = lmm.findH2(nGridH2=100)
            h2 = opt['h2']
            assert self.delta is None, "either findh2 or set delta"
        else:
            h2 = 0.0
            assert not self.delta is None
            logging.info("using externally provided delta")

        res = lmm.nLLeval(h2=h2, delta=self.delta, dof=None, scale=1.0, penalty=0.0, snps=self.test_snps)
        
        
        chi2stats = res['beta']*res['beta']/res['variance_beta']
        
        self.p_values = stats.chi2.sf(chi2stats,1)[:,0]
        self.p_values_F = stats.f.sf(chi2stats,1,G.shape[0]-(lmm.linreg.D+1))[:,0]#note that G.shape is the number of individuals
        self.p_idx = np.argsort(self.p_values)        
        self.sorted_p_values = self.p_values[self.p_idx]

        self.p_idx_F = np.argsort(self.p_values_F)
        self.sorted_p_values_F = self.p_values_F[self.p_idx_F]