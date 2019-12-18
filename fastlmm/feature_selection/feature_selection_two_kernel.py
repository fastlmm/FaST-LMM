"""
stand-alone module to perform feature selection with two kernels

This method selects SNPs for a foreground kernel, given a background kernel.
The background kernel is obtained from all SNPs and is intended to capture
population structure, as well as family structure.

Selection is performed by first ranking SNPs based on a LMM using
the full kernel according to their univariate association with the phenotype
and then choosing a number to cut off using cross-validation.
"""

from __future__ import absolute_import
from __future__ import print_function
import time
import logging

import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from fastlmm.association.LocoGwas import FastGwas
import fastlmm.inference.linear_regression as lin_reg
from six.moves import range



class FeatureSelectionInSample(object):

    def __init__(self, n_folds=5, measure="mse", n_k_grid=7, max_log_k=10, order_by_lmm=False, random_state=None, debug=False):
        """set up two kernel feature selection
    
        Parameters
        ----------
        n_folds : int
            Number of repeats for shuffle split

        measure : string ("mse" or "ll")  
            Criterion to be used to measure predition accuracy

        n_k_grid : int
            Number of k-values to search

        max_log_k : int
            Maximum search value for number of snps in log2-space (i.e. 2^max_log_k)

        order_by_lmm : bool
            If set to true, an LMM based on the full kernel will be used to rank SNPs,
            otherwise LR is used

        random_state : int
            Random state for splits of the data

        debug : bool
            Runs some additional checks if set to True (useful for unit tests)

        Returns
        -------
        list of return values (one for each input argument)
        """

        self.measure = measure
        self.order_by_lmm = order_by_lmm

        self.n_folds = n_folds
        self.n_k_grid = n_k_grid

        self.mse = np.zeros((self.n_folds, self.n_k_grid))
        self.ll = np.zeros((self.n_folds, self.n_k_grid))

        self.grid_k = [int(k) for k in np.logspace(0, max_log_k, base=2, num=self.n_k_grid, endpoint=True)]
        print(self.grid_k)
        self.random_state = random_state
        self.mixes = np.zeros((self.n_folds, self.n_k_grid))
        self.h2 = np.zeros((self.n_folds, self.n_k_grid))
        self.deltas = np.zeros((self.n_folds, self.n_k_grid))

        self.debug = False

    def run_select(self, G0, G_bg, y, cov=None):
        """set up two kernel feature selection
    
        Parameters
        ----------
        G0 : numpy array of shape (num_ind, num_snps)
            Data matrix from which foreground snps will be selected

        G0_bg : numpy array of shape (num_ind, num_snps)
            Data matrix containing background snps on which will be conditioned

        y : numpy vector of shape (num_ind, )
            Vector of phenotypes

        cov : numpy array of shape (num_ind, num_covariates) or None
            Covariates to be used as fixed effects

        Returns
        -------
        best_k, feat_idx, best_mix, best_delta: tuple(int, np.array(int), float, float)
            best_k is the best number of SNPs selected,
            feat_idx is a np.array of integers denoting the indices of these snps,
            best_mix is the best mixing coefficient between foreground and background kernel,
            best_delta is the best regularization coefficient
        """

        num_ind = len(y)

        if cov is None:
            cov = np.ones((num_ind,1))
        else:
            logging.info("normalizing covariates")
            cov = cov.copy()
            cov = 1./np.sqrt((cov**2).sum() / float(cov.shape[0])) * cov
        cov.flags.writeable = False
        
        # normalize to diag(K) = N
        norm_factor = 1./np.sqrt((G_bg**2).sum() / float(G_bg.shape[0]))

        # we copy in case G and G_bg are pointing to the same object
        G_bg = norm_factor * G_bg
       
        K_bg_full = G_bg.dot(G_bg.T)
        K_bg_full.flags.writeable = False
        
        # some asserts
        np.testing.assert_almost_equal(sum(np.diag(K_bg_full)), G_bg.shape[0])
        if self.debug:
            norm_factor_check = 1./np.sqrt(G_bg.shape[1])
            np.testing.assert_array_almost_equal(norm_factor, norm_factor_check, decimal=1)
            

        for kfold_idx, (train_idx, test_idx) in enumerate(KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True).split(list(range(num_ind)))):

            t0 = time.time()
            logging.info("running fold: %i" % kfold_idx)

            y_train = y.take(train_idx, axis=0)
            y_test = y.take(test_idx, axis=0)
            G0_train = G0.take(train_idx, axis=0)
            G0_test = G0.take(test_idx, axis=0)

            G_bg_train = G_bg.take(train_idx, axis=0)
            G_bg_test = G_bg.take(test_idx, axis=0)

            cov_train = cov.take(train_idx, axis=0)
            cov_test = cov.take(test_idx, axis=0)

            # write protect data
            y_train.flags.writeable = False
            y_test.flags.writeable = False
            G0_train.flags.writeable = False
            G0_test.flags.writeable = False
            G_bg_train.flags.writeable = False
            G_bg_test.flags.writeable = False
            cov_train.flags.writeable = False
            cov_test.flags.writeable = False

            # precompute background kernel
            K_bg_train = K_bg_full.take(train_idx, axis=0).take(train_idx, axis=1) 
            K_bg_train.flags.writeable = False

            if self.measure != "mse":
                K_bg_test = K_bg_full.take(test_idx, axis=0).take(test_idx, axis=1)
                K_bg_test.flags.writeable = False

            # rank features
            if self.order_by_lmm:
                logging.info("using linear mixed model to rank features")
                t0 = time.time()
                gwas = FastGwas(G_bg_train, G0_train, y_train, delta=None, train_pcs=None, mixing=0.0, cov=cov_train)
                gwas.run_gwas()
                _pval = gwas.p_values
                logging.info("time taken: %s" % (str(time.time()-t0)))
            else:
                logging.info("using linear regression to rank features")
                _F,_pval = lin_reg.f_regression_block(lin_reg.f_regression_cov_alt, G0_train, y_train, blocksize=10000, C=cov_train)

            feat_idx = np.argsort(_pval)
            
            for k_idx, max_k in enumerate(self.grid_k):

                feat_idx_subset = feat_idx[0:max_k]
                G_fs_train = G0_train.take(feat_idx_subset, axis=1)
                G_fs_test = G0_test.take(feat_idx_subset, axis=1)

                # normalize to sum(diag)=N
                norm_factor = 1./np.sqrt((G_fs_train**2).sum() / float(G_fs_train.shape[0]))

                G_fs_train *= norm_factor
                G_fs_test *= norm_factor
                                
                G_fs_train.flags.writeable = False
                G_fs_test.flags.writeable = False

                # asserts
                if self.debug:
                    norm_factor_check = 1.0 / np.sqrt(max_k)
                    np.testing.assert_array_almost_equal(norm_factor, norm_factor_check, decimal=1)
                    np.testing.assert_almost_equal(sum(np.diag(G_fs_train.dot(G_fs_train.T))), G_fs_train.shape[0])

                logging.info("k: %i" % (max_k))

                # use LMM
                from fastlmm.inference.lmm_cov import LMM as fastLMM

                if G_bg_train.shape[1] <= G_bg_train.shape[0]:
                    lmm = fastLMM(X=cov_train, Y=y_train[:,np.newaxis], G=G_bg_train)
                else:
                    lmm = fastLMM(X=cov_train, Y=y_train[:,np.newaxis], K=K_bg_train)

                W = G_fs_train.copy()
                UGup,UUGup = lmm.rotate(W)
                
                i_up = np.zeros((G_fs_train.shape[1]), dtype=np.bool)
                i_G1 = np.ones((G_fs_train.shape[1]), dtype=np.bool)
                t0 = time.time()
                res = lmm.findH2_2K(nGridH2=10, minH2=0.0, maxH2=0.99999, i_up=i_up, i_G1=i_G1, UW=UGup, UUW=UUGup)
                logging.info("time taken for k=%i: %s" % (max_k, str(time.time()-t0)))
                
                # recover a2 from alternate parameterization
                a2 = res["h2_1"] / float(res["h2"] + res["h2_1"])
                h2 = res["h2"] + res["h2_1"]
                delta = (1-h2) / h2
                #res_cov = res


                # do final prediction using lmm.py
                from fastlmm.inference import LMM
                lmm = LMM(forcefullrank=False)
                lmm.setG(G0=G_bg_train, G1=G_fs_train, a2=a2)
                lmm.setX(cov_train)
                lmm.sety(y_train)

                # we take an additional step to estimate betas on covariates (not given from new model)
                res = lmm.nLLeval(delta=delta, REML=True)
                
                # predict on test set
                lmm.setTestData(Xstar=cov_test, G0star=G_bg_test, G1star=G_fs_test)
                out = lmm.predictMean(beta=res["beta"], delta=delta)

                mse = mean_squared_error(y_test, out)
                logging.info("mse: %f" % (mse))

                self.mse[kfold_idx, k_idx] = mse

                self.mixes[kfold_idx, k_idx] = a2
                self.deltas[kfold_idx, k_idx] = delta

                if self.measure != "mse":
                    K_test_test = a2 * G_fs_test.dot(G_fs_test.T) + (1.0-a2) * K_bg_test 
                    ll = lmm.nLLeval_test(y_test, res["beta"], sigma2=res["sigma2"], delta=delta, Kstar_star=K_test_test, robust=True)

                    if self.debug:
                        ll2 = lmm.nLLeval_test(y_test, res["beta"], sigma2=res["sigma2"], delta=delta, Kstar_star=None, robust=True)
                        np.testing.assert_almost_equal(ll, ll2, decimal=4)

                    logging.info("ll: %f" % (ll))
                    self.ll[kfold_idx, k_idx]  = ll
                    

            logging.info("time taken for fold: %s" % str(time.time()-t0))
        

        best_k, best_mix, best_delta = self.select_best_k()

        logging.info("best_k: %i, best_mix: %f, best_delta: %f" % (best_k, best_mix, best_delta))

        # final scan 
        if self.order_by_lmm:
            logging.info("final scan using LMM")
            gwas = FastGwas(G_bg, G0, y, delta=None, train_pcs=None, mixing=0.0, cov=cov)
            gwas.run_gwas()
            _pval = gwas.p_values
            feat_idx = np.argsort(_pval)[0:best_k]
        else:
            logging.info("final scan using LR")
            _F,_pval = lin_reg.f_regression_block(lin_reg.f_regression_cov_alt, G0, y, C=cov, blocksize=10000)
        
        logging.info("number of snps selected: %i" % (best_k))

        return best_k, feat_idx, best_mix, best_delta


    def select_best_k(self):
        """after running cross-vlidation, choose best number of snps

        Returns
        -------
        best_k, best_mix, best_delta: tuple(int, float, float)
            best_k is the best number of SNPs selected,
            best_mix is the best mixing coefficient between foreground and background kernel,
            best_delta is the best regularization coefficient
        """

        perf = self.mse

        if self.measure == "ll":
            logging.info("using nLL to select best_k")
            perf = self.ll
        else:
            logging.info("using MSE to select best_k")

        min_ = perf.mean(axis=0)
        arg_min_k = np.argmin(min_)


        best_k = self.grid_k[arg_min_k]
        best_mix = np.median(self.mixes[:,arg_min_k])
        best_delta = np.median(self.deltas[:,arg_min_k])
        return best_k, best_mix, best_delta


    def plot_results(self, measure="mse"):
        """
        visualize trained model, either measure="ll" or measure="mse"
        """
        import pylab
        if measure == "ll":
            pylab.plot(self.grid_k, self.ll.mean(axis=0).T, "-x", label=self.grid_k)
            pylab.ylabel("-ll")
        else:
            pylab.plot(self.grid_k, self.mse.mean(axis=0).T, "-x", label=self.grid_k)
            pylab.ylabel("mse")
        #pylab.yscale("log")
        #pylab.xscale("log")
        pylab.xlabel("num features in foreground")
        
        pylab.grid(True)
        pylab.title(measure)
        #pylab.legend()
        pylab.show()
    
