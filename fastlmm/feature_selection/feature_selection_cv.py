"""
Created on 2013-07-28
@author: Christian Widmer <chris@shogun-toolbox.org>
@summary: Module for feature selection strategies using efficient caching where possible

"""

# std modules
from __future__ import absolute_import
from __future__ import print_function
from collections import defaultdict
import gzip
import bz2
import six.moves.cPickle
import time
import os 
import gc
import subprocess
import sys
import logging

# common modules
import scipy as sp
import numpy as np
import pandas as pd

# sklearn
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit
from sklearn.datasets import load_boston, load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn import model_selection
from sklearn.decomposition import KernelPCA

# project
from pysnptools.snpreader import Bed
import pysnptools.util as pstutil
import pysnptools.util.pheno as pstpheno
import fastlmm.util.util as util 
import fastlmm.util.preprocess as up
import fastlmm.inference as inference
import fastlmm.inference.linear_regression as lin_reg
from fastlmm.feature_selection import PerformSelectionDistributable as psd
from pysnptools.util.mapreduce1.runner import *
from pysnptools.standardizer import Unit
import pysnptools.snpreader as sr
from six.moves import range

class FeatureSelectionStrategy(object):

    def __init__(self, snpreader, pheno_fn, num_folds, test_size=0.1, cov_fn=None, num_snps_in_memory=100000,
                 random_state=None, log=None, offset=True, num_pcs=0, interpolate_delta=False, mpheno = 0, standardizer=Unit(),count_A1=None):
    
        """Set up Feature selection strategy
        ----------

        snpreader : str or snpreader
            File name of binary SNP file or a snpreader.

        pheno_fn : str
            File name of phenotype file

        num_folds : int
            Number of folds in k-fold cross-validation

        test_size : float, default=0.1
            Fraction of samples to use as test set (train_size = 1-test_size)

        cov_fn : str, optional, default=None
            File name of covariates file

        num_snps_in_memory: int, optional, default=100000
            Number of SNPs to keep in memory at a time. Setting this higher than the largest k
            will dramatically increase speed at the cost of higher memory use.

        random_state : int, default=None
            Seed to use for random number generation (e.g. random splits)

        log : Level of log messages, defaults=None (don't change)
            e.g. logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO

        offset : bool, default=True
            Adds offset to the covariates specified in cov_fn, if necessary

        num_pcs : int, default=0
            Number of principle components to be included as fixed effects.
            If num_pcs>0, a PCA will be computed as preprocessing.

        interpolate_delta : bool, default=False
            Interpolate delta around optimum with parabola (for best k).

        mpheno : int, default=0
            Column id of phenotype

        standardizer: a standandizer-like object such as Unit() or Beta(1,25), default=Unit()

        count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
             alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.


        """
        self._ran_once = False
 
        # data file names
        self.snpreader = snpreader
        self.count_A1 = count_A1
        if isinstance(self.snpreader, str):
            self.snpreader = Bed(self.snpreader,count_A1=count_A1)
        #!!test speed of new vs old
        #!!make all readers take optional file extension

        self.pheno_fn = pheno_fn
        self.cov_fn = cov_fn

        # data fields
        self.G = None
        self.y = None
        self.X = None

        # flags
        self.num_folds = num_folds
        self.test_size = test_size
        self.random_state = random_state
        self.offset = offset
        self.num_pcs = num_pcs
        self.pcs = None
        self.interpolate_delta = interpolate_delta
        self.mpheno = mpheno
        self.standardizer = standardizer

        # efficiency
        self.num_snps_in_memory = num_snps_in_memory
        self.blocksize = 1000
        self.biggest_k = None

        if log is not None:
            logger.setLevel(log)


    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True

        # load data
        self.load_data()

        # precompute kernel on all snps if needed
        if self.num_pcs > 0 or self.biggest_k >= self.snpreader.sid_count:
            from pysnptools.standardizer import Identity
            self.K = self.G.kernel()
            self.K.flags.writeable = False

        # optionally perform pca
        if self.num_pcs > 0:
            self.perform_pca()

    def load_covariates(self, pheno):
        if self.cov_fn == None:
            cov_iid = pheno['iid']
            X = np.ones((len(cov_iid), 1))
        else:
            cov = pstpheno.loadPhen(self.cov_fn)
            X = cov['vals']
            cov_iid = cov['iid']
            # add bias column if not present - #!! LATER -- Bug? should this test be done after intersection in case removing an iid makes it constant?
            if self.offset and sp.all(X.std(0)!=0):
                offset = sp.ones((len(X),1))
                self.X = sp.hstack((X, offset))
        return X, cov_iid


    def load_data(self):
        """load data
        """

        
        tt0 = time.time()
        logging.info("loading data...")

        if self.num_snps_in_memory <= self.snpreader.iid_count : raise Exception("Expect self.num_snps_in_memory, {0} > self.snpreader.iid_count, {1}".format(self.num_snps_in_memory, self.total_num_ind))

        self.sid = pd.Series(self.snpreader.sid)

        # load phenotype
        pheno = pstpheno.loadOnePhen(self.pheno_fn,self.mpheno, vectorize=True)
        self.ind_iid = pheno['iid'] #!!LATER: bug? It looks like we record the pre-intersect iids only to write out the pcs later? Why?

        # load covariates
        self.X, cov_iid = self.load_covariates(pheno)

        # Set up the snps
        # G is the standardized snps. The GClass.factory will either load them into memory or will note their file and read them as needed.
        self.G = GClass.factory(self.snpreader, self.num_snps_in_memory, self.standardizer, self.blocksize,count_A1=self.count_A1)

        #!!LATER Should we give preference to self.G since reordering it is the most expensive?
        (self.y, yiid), (self.X, xiid), self.G = pstutil.intersect_apply([(pheno['vals'], pheno['iid']), (self.X, cov_iid), self.G], sort_by_dataset=False)

        # make sure input data isn't modified
        self.X.flags.writeable = False
        self.y.flags.writeable = False

        logging.info("...done. Loading time %.2f s" % (float(time.time() - tt0)))


    def perform_pca(self):
        """consider principle components as covariates, will be appended to self.X

        num_pcs : int
            Number of principle components to use as covariates

        
        K = self._centerer.fit_transform(K)

        # compute eigenvectors
        if self.eigen_solver == 'auto':
            if K.shape[0] > 200 and n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver

        if eigen_solver == 'dense':
            self.lambdas_, self.alphas_ = linalg.eigh(
                K, eigvals=(K.shape[0] - n_components, K.shape[0] - 1))
        elif eigen_solver == 'arpack':
            self.lambdas_, self.alphas_ = eigsh(K, n_components,
                                                which="LA",
                                                tol=self.tol,
                                                maxiter=self.max_iter)

        # sort eigenvectors in descending order
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]

        # remove eigenvectors with a zero eigenvalue
        if self.remove_zero_eig or self.n_components is None:
            self.alphas_ = self.alphas_[:, self.lambdas_ > 0]
            self.lambdas_ = self.lambdas_[self.lambdas_ > 0]

        X_transformed = self.alphas_ * np.sqrt(self.lambdas_)

        """
        #TODO: implement numerics code directly, based on above template

        logging.info("performing PCA, keeping %i principle components" % (self.num_pcs))
        tt0 = time.time()
        if False:
            pca = KernelPCA(n_components=self.num_pcs)
            pca._fit_transform(self.K)
            self.pcs = pca.alphas_ * np.sqrt(pca.lambdas_)
        else:
            import scipy.linalg as la
            [s,u]=la.eigh(self.K)
            s=s[::-1]
            u=u[:,::-1]
            self.pcs = u[:,0:self.num_pcs]
        assert self.pcs.shape[1] == self.num_pcs

        self.X = sp.hstack((self.X, self.pcs))  

        logging.info("...done. PCA time %.2f s" % (float(time.time() - tt0)))


    def setup_linear_regression(self, max_k, start=0, stop=None):
        """precompute univariate ranking for each split

        max_k : int
            Maximum number of SNPs to store for precomputation.
            SNPs will be sorted ascendingly by p-value and 
            at most max_k features will be kept.
        
        """

        self.run_once()

        # set up splitting strategy
        kf = ShuffleSplit(n_splits=self.num_folds, test_size=self.test_size, random_state=self.random_state).split(list(range(len(self.y))))

        fold_idx = start -1
        for (train_idx, test_idx) in islice(kf,start,stop):
            fold_idx += 1

            logging.info("processing split {0}".format(fold_idx))

            tt0 = time.time()
            fold_data = {}
            fold_data["train_idx"] = train_idx
            fold_data["test_idx"] = test_idx

            # set up data
            ##############################
            fold_data["X_train"] = self.X[train_idx,:]
            fold_data["X_test"] = self.X[test_idx,:]
            fold_data["y_train"] = self.y[train_idx]
            fold_data["y_test"] = self.y[test_idx]

            logging.info("indexing time over samples %.2f s" % (float(time.time() - tt0)))

            # feature selection
            ##############################
            # Note: if you use covariates here, they will be regressed out of G_train, but not of G_test!
            tt0 = time.time()

            fold_data["G_train"] = self.G[train_idx,:]
            fold_data["G_test"] = self.G[test_idx,:]
            _F,_pval = fold_data["G_train"].lin_reg(fold_data["y_train"], fold_data["X_train"])

            logging.info("lin_regr time %.2f s" % (float(time.time() - tt0)))

            tt0 = time.time()
            feat_idx = np.argsort(_pval)
            fold_data["feat_idx"] = feat_idx
            
            # re-order SNPs (and cut to max num)
            ##############################
            fold_data["G_train"] = fold_data["G_train"][:,feat_idx[0:max_k]]
            fold_data["G_test"] = fold_data["G_test"][:,feat_idx[0:max_k]]

            # set to read-only to make sure data isn't modified
            ##############################
            fold_data["X_train"].flags.writeable = False
            fold_data["X_test"].flags.writeable = False
            fold_data["y_train"].flags.writeable = False
            fold_data["y_test"].flags.writeable = False

            logging.info("indexing time over SNPs %.2f s" % (float(time.time() - tt0)))


            yield fold_data


    #TODO: once the functionality is fixed refactor
    def perform_selection(self, k_values, delta_values, strategy="lmm_full_cv", output_prefix=None, select_by_ll=False, runner=Local(),penalty=0.0,create_pdf=True):
        """Perform feature selection

        k_values : array-like, shape = [n_steps_k]
            Array of k values to test

        delta_values : array-like, shape = [n_steps_delta]
            Array of delta values to test

        strategy : {'lmm_full_cv', 'insample_cv'}
            Strategy to perform feature selection:

            - 'lmm_full_cv' perform cross-validation over grid of k and delta using LMM
              
            - 'insample_cv' perform cross-validation over grid of k, estimate delta in sample
              using maximum likelihood.

        output_prefix : str, optional, default=None
            Prefix for output files

        select_by_ll : bool, default=False
            If set to True, negative log-likelihood will be used to select best k and delta


        Returns
        -------
        best_k : int
            best subset size k

        best_delta : float
            best regularization parameter delta for ridge regression

        best_obj : float
            best objective at optimum (default MSE, nLL if select_by_ll flag is set), 

        best_snps : list[str]
            list of ids of best snps (univariate selection done on whole data set using best_k, best_delta)

        """

        self.biggest_k = max(k_values)
        
        if (strategy!="lmm_full_cv") and (strategy!="insample_cv"):
            logging.warning("strategies other than lmm_full_cv and insample_cv are experimental!")
            raise Exception("strategies other than lmm_full_cv and insample_cv are experimental!")

        perform_selection_distributable = psd.PerformSelectionDistributable(self, k_values, delta_values, strategy, output_prefix, select_by_ll, penalty=penalty,create_pdf=create_pdf)
        result = runner.run(perform_selection_distributable)
        return result

    def copyinputs(self, copier):
        copier.input(self.snpreader)
        copier.input(self.pheno_fn)
        if (self.cov_fn is not None):
            copier.input(self.cov_fn)
        
    def copyoutputs(self, copier):
        pass

    def reduce_result(self, loss_cv, k_values, delta_values, strategy, output_prefix,best_delta_for_k, label="mse", create_pdf=True):
        """
        turn cross-validation results into results
        """
        #self.run_once() #Don't need and saves time

        # average over splits
        average_loss = np.array(loss_cv).mean(axis=0)
        #average_loss = np.vstack(loss_cv).mean(axis=0)
        best_ln_delta_interp, best_obj_interp, best_delta_interp = (None,None,None)

        # reconstruct results
        if strategy == "lmm_full_cv":
            # save cv scores
            if output_prefix != None:
                split_idx = ["mean"]*len(k_values)
                for idx in range(len(loss_cv)):
                    split_idx.extend([idx]*loss_cv[idx].shape[0])
                                
                stacked_result = np.vstack(loss_cv)
                stacked_result = np.vstack((average_loss, stacked_result))
                
                out_fn = output_prefix + "_" + label  + ".csv"
                cols = pd.MultiIndex.from_arrays([split_idx, k_values*(self.num_folds+1)], names=['split_id','k_value'])
                df = pd.DataFrame(stacked_result, columns=delta_values, index=cols)
                pstutil.create_directory_if_necessary(out_fn)
                df.to_csv(out_fn)
            
            # make sure delta is not at the boundary for any k
            assert average_loss.shape[0] == len(k_values)
            for k_idx in range(average_loss.shape[0]):
                tmp_idx = np.argmin(average_loss[k_idx])
                
                if tmp_idx == 0 or tmp_idx == len(delta_values)-1:
                    logging.warning("(select by %s): ln_delta for k=%i is at the boundary (idx=%i) of defined delta grid" % (label, k_values[k_idx], tmp_idx))
            
            best_k_idx, best_delta_idx = np.unravel_index(average_loss.argmin(), average_loss.shape)
            best_k, best_delta = k_values[best_k_idx], delta_values[best_delta_idx]
            best_obj = average_loss[best_k_idx, best_delta_idx]
            best_ln_delta = np.log(best_delta)
            best_str = "best: k=%i, ln_d=%.1f, obj=%.2f" % (best_k, best_ln_delta, best_obj)
            
            # fit parabola to 3 points in logspace
            if self.interpolate_delta:
                if best_delta_idx!=0 and best_delta_idx!=len(delta_values)-1:
                    log_deltas = [np.log(d) for d in delta_values[best_delta_idx-1:best_delta_idx+2]]
                    error_3pt = average_loss[best_k_idx, best_delta_idx-1:best_delta_idx+2]
                    
                    best_ln_delta_interp, best_obj_interp = self.fit_parabola(log_deltas, error_3pt, output_prefix=None)
                    best_delta_interp = sp.exp(best_ln_delta_interp)
                    best_str += ", ln_d_interp=%.2f" % (best_ln_delta_interp)
                    logging.info("best interpolated ln_delta {0}".format(best_ln_delta_interp))
                else:
                    logging.warning("(select by %s): best ln_delta for all k is at the boundary (idx=%i) of search grid, please consider a larger grid" % (label, best_delta_idx))
                    #if output_prefix != None:
                        #create a size-zero file so that the cluster will aways have something to copy
                        #plot_fn=output_prefix+"_parabola.pdf"
                        #pstutil.create_directory_if_necessary(plot_fn)
                        #open(plot_fn, "w").close()

            # save cv scores
            if create_pdf and (output_prefix != None):
                # visualize results
                import pylab
                pylab.figure()
                ax = pylab.subplot(111)
                try:
                    for delta_idx, delta in enumerate(delta_values):
                        ln_delta = sp.log(delta)
                        ax.semilogx(k_values, average_loss[:,delta_idx], "-x", label="ln_d=%.1f" % (ln_delta))

                    # Shrink current axis by 20%
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                    #TODO: this assumes the k_values are sorted:
                    pylab.ylim(ymax=average_loss[0].max() + abs(average_loss[0].max())*0.05 )
                    if k_values[0] != 0: logging.warning("Expect the first k value to be zero") #!!move this change earlier
                    for i in range(len(k_values)):
                        if k_values[i] == 0:
                            ax.axhline(average_loss[i].max(), color = 'green')
                            mymin = average_loss.min() 
                            mymax = average_loss[i].max()
                            diff = (mymax-mymin)*0.05
                            pylab.ylim([mymin-diff,mymax+diff])                

                    # Put a legend to the right of the current axis
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                    pylab.title(best_str)
                    pylab.ylabel(label)
                    pylab.xlabel("k")
                    pylab.grid(True)
                    #pylab.show()
                except:
                    pass
                xval_fn = output_prefix + "_xval_%s.pdf" % label
                pstutil.create_directory_if_necessary(xval_fn)
                pylab.savefig(xval_fn)
        elif strategy == "insample_cv":
            best_k_idx = average_loss.argmin()
            best_k = k_values[best_k_idx]
            best_obj = average_loss[best_k_idx]

            # check if unique over folds
            delta_array = np.array(best_delta_for_k)
            unique_deltas_for_k = set(delta_array[:,best_k_idx])
            if len(unique_deltas_for_k) > 1:
                logging.warning("ambiguous choice of delta for k: {0} {1}".format(best_k, unique_deltas_for_k))

            best_delta = np.median(delta_array[:,best_k_idx])

            best_str = "best k=%i, best delta=%.2f" % (best_k, best_delta)
            logging.info(best_str)
            if output_prefix != None:
                split_idx = ["mean"]*len(k_values)
                for idx in range(len(loss_cv)):
                    split_idx.extend([idx]*loss_cv[idx].shape[0])
                                
                stacked_result = np.vstack(loss_cv)
                stacked_result = np.vstack((average_loss, stacked_result))
                out_fn = output_prefix + "_" + label  + ".csv"
                cols = pd.MultiIndex.from_arrays([split_idx, k_values*(self.num_folds+1)], names=['split_id','k_value'])
                print("Christoph: bug, this is a quick fix that runs but may write out wrong results")
                df = pd.DataFrame(stacked_result.flatten()[:, None], columns=[label], index=cols)
                pstutil.create_directory_if_necessary(out_fn)
                df.to_csv(out_fn)
            if create_pdf and (output_prefix != None):
                # visualize results
                import pylab
                pylab.figure()
                ax = pylab.subplot(111)
                try:
                    ax.semilogx(k_values, average_loss, "-x", label="loo")

                    # shrink current axis by 20%
                    box = ax.get_position()
                    #TODO: this assumes the k_values are sorted:
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    pylab.ylim(ymax=average_loss[0].max() + abs(average_loss[0].max())*0.05 )
                    if k_values[0] != 0: logging.warning("Expect the first k value to be zero") #!!move this change earlier
                    for i in range(len(k_values)):
                        if k_values[i] == 0:
                            ax.axhline(average_loss[i].max(), color = 'green')
                            mymin =average_loss.min() 
                            mymax = average_loss[i].max()
                            diff = (mymax-mymin)*0.05
                            pylab.ylim([mymin-diff,mymax+diff])
                    # Put a legend to the right of the current axis
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    pylab.title(best_str)
                    pylab.ylabel(label)
                    pylab.xlabel("k")
                    pylab.grid(True)
                except:
                    pass
                plot_fn = output_prefix + "_xval_%s.pdf"%label
                pstutil.create_directory_if_necessary(plot_fn)
                pylab.savefig(plot_fn)
        else:
            raise NotImplementedError(strategy)
        return best_k, best_delta, best_obj, best_delta_interp, best_obj_interp

    def pcs_filename(self,output_prefix):
        return "{0}_k_pcs.txt".format(output_prefix)

    def linreg_entire_dataset(self, output_prefix):
        self.run_once()

        # We do a little extra work here. We write the pcs out here because they will be known here and we know this method will only be run
        # once even on a big cluster jobs
        if output_prefix is not None and self.num_pcs > 0:
            filename = self.pcs_filename(output_prefix)
            import fastlmm.util.util as ut        
            ut.write_plink_covariates(self.ind_iid,self.pcs,filename)

        logging.info("performing final scan through entire data set")
        tt0 = time.time()

        _F,_pval = self.G.lin_reg(self.y, self.X)

        feat_idx = np.argsort(_pval)

        _pval_feat_idx = _pval[feat_idx]
        sid_feat_idx = self.sid[feat_idx]
        lingreg_results = (_pval[feat_idx],self.sid[feat_idx])
        logging.info("fin_scan time %.2f s" % (float(time.time() - tt0)))

        return lingreg_results



    def final_scan(self, best_k, lingreg_results):
        """Scan through entire dataset one last time and report top best_k SNP ids
        ----------

        best_k : int
            Number of SNPs to report

        Returns
        -------
        sorted_sid : list, len = best_k
            List of SNP ids, sorted by p-value

        sorted_pval : list, len = best_k
            List of p-values
        """


        _pval_feat_idx,sid_feat_idx = lingreg_results
        sorted_pval = _pval_feat_idx[0:best_k]
        sorted_sid = sid_feat_idx[0:best_k]
        return sorted_sid, sorted_pval


    def fit_parabola(self, deltas, perf, output_prefix=None, create_pdf=True):
        """
        for best k, fit parabola to 3 points in delta dimension and determine optimum accordingly,
        assume convex function
        """

        #self.run_once() - don't call this because we don't need snp data loaded

        assert len(perf) == len(deltas)
        assert perf[1] <= perf[0]
        assert perf[1] <= perf[2]
        assert deltas[0] < deltas[1] < deltas[2]

        coef = np.polyfit(deltas, perf, 2)

        xfit = np.linspace(min(deltas), max(deltas), num=100)
        yfit =  np.polyval(coef, xfit)

        best_idx = np.argmin(yfit)
        best_delta = xfit[best_idx]
        best_obj = yfit[best_idx]

        #if create_pdf and (output_prefix!=None):
        #    pylab.figure()
        #    pylab.plot(deltas, perf, '.')
        #    pylab.plot(xfit, yfit, '-')
        #    pylab.grid(True)
        #    plot_fn = output_prefix + "_parabola.pdf"
        #    pstutil.create_directory_if_necessary(plot_fn)
        #    pylab.savefig(plot_fn)

        return best_delta, best_obj


def f_regression_block_load(fun, snpreader, standardizer, y, ind_idx=None, blocksize=10000, **args):
    """
    runs f_regression for each block separately (saves memory).

    -------------------------
    fun        : method that returns statistics,pval
    snpreader  : reader object
    y          : array of shape(n_samples).
    blocksize  : number of SNPs per block
    """

    logging.info("running linear regression in blocks")

    num_snps = snpreader.sid_count

    #This needs testing
    #if blocksize==None:
    #    X = snpreader.read().standardize(standardizer).val
    #    if ind_idx is not None:
    #        X = X[ind_idx,:]
    #    return fun(X,y,**args)

    idx_start = 0
    idx_stop = blocksize
     
    pval = np.zeros(num_snps)
    stats = np.zeros(num_snps)

    for start in range(0,snpreader.sid_count,blocksize):
        partialX = snpreader[:,start:start+blocksize].read().standardize(standardizer).val
        if ind_idx is not None:
            partialX = partialX[ind_idx,:]
        stats[idx_start:idx_stop], pval[idx_start:idx_stop] = fun(partialX, y, **args)
        idx_start = idx_stop
        idx_stop += blocksize
        if idx_stop>num_snps:
            idx_stop = num_snps


    return stats, pval


        
def load_snp_data(snpreader, pheno_fn, cov_fn=None, offset=True, mpheno=0, standardizer=Unit()):
    """Load plink files
    ----------

    snpreader : snpreader object
        object to read in binary SNP file

    pheno_fn : str
        File name of phenotype file

    cov_fn : str
        File name of covariates file

    offset : bool, default=True
        Adds offset to the covariates specified in cov_fn, if neccesssary


    Returns
    -------
    G : array, shape = [n_samples, n_features]
        SNP matrix

    X : array, shape = [n_samples, n_covariates]
        Matrix of covariates (e.g. age, gender)

    y : array, shape = [n_samples]
        Phenotype (target) vector

    """
    
    #TODO: completely remove this
    pheno = pstpheno.loadOnePhen(pheno_fn,mpheno, vectorize=True)
    geno = snpreader.read(order='C').standardize(standardizer)

    # sanity check
    #assert np.testing.assert_array_equal(ind_iid, pheno['iid'][indarr[:,0]])

    # load covariates or generate vector of ones (for bias)
    if cov_fn == None:
        cov = {'vals': np.ones((len(pheno['iid']), 1)), 'iid':pheno['iid']}
    else:
        cov = pstpheno.loadPhen(cov_fn)

    (y, yiid), G, (X, xiid) = pstutil.intersect_apply([(pheno['vals'],pheno['iid']), geno, (cov['vals'],cov['iid'])], sort_by_dataset=False)
    G = G.read(order='C', view_ok=True)

    # add bias column if not present
    if offset and sp.all(X.std(0)!=0):
        offset = sp.ones((len(indarr),1))
        X = sp.hstack((X,offset))  
        
    return G, X, y
 

from pysnptools.standardizer import Unit
from pysnptools.standardizer import Identity
from pysnptools.standardizer import Beta
from pysnptools.standardizer import BySqrtSidCount
from pysnptools.standardizer import BySidCount
def factory(s):
    s = s.capitalize()
    if s == "Unit" or s=="Unit()":
        return Unit()

    if s == "Identity" or s=="Identity()":
        return Identity()

    if s == "BySqrtSidCount" or s=="BySqrtSidCount()":
        return BySqrtSidCount()

    if s == "BySidCount" or s=="BySidCount()":
        return BySidCount()

    if s=="Beta":
        return Beta()

    if s.startswith("Beta("):
        standardizer = eval(s)
        return standardizer


def main():
    """
    command line interface to fastLMM-select
    """

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required input
    parser.add_argument('snpreader', help='bed file name or bed file reader', type=str)
    parser.add_argument('pheno_fn', help='path to phenotype file', type=str)

    # optional arguments
    parser.add_argument('--k_values', help='List of snp counts to try. Can use "all" as a value.', type=str, default="1,2,4,8,16,32,64,128,256,1024,2048,all")
    parser.add_argument('--ln_delta_values', help='List of ln_delta values to try.', type=str, default="-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument('--cov_fn', help='path to covariates file', type=str, default=None)
    parser.add_argument('--seed', help='random seed to use for splitting data sets', type=int, default=None)
    parser.add_argument('--log', help='flag to control verbosity of logging', type=str, default="WARNING")
    parser.add_argument('--strategy', help='feature selection strategy to use {lmm_full_cv, insample_cv}', type=str, default="lmm_full_cv")
    parser.add_argument('--num_splits', help='number of splits to use in cross-validation', type=int, default=10)
    parser.add_argument('--num_pcs', help='number of principle components to include as fixed effects (PCA will be performed)', type=int, default=0)
    parser.add_argument('--test_size', help='fraction of examples to use as test set (train_size=1-test_size)', type=float, default=0.1)
    parser.add_argument('--select_by_ll', help=' if set to True, negative log-likelihood will be used to select best k and delta (instead of MSE)', type=bool, default=False)
    parser.add_argument('--interpolate_delta', help=' if set to True, will interpolate delta around optimum with parabola (for best k)', type=bool, default=False)
    parser.add_argument('--standardizer', help='"unit", or "beta(a,b)" where a and b are positive numbers, or "beta" which is the same as "beta(1,25)"', type=str, default="unit")
    parser.add_argument('--mpheno', help="index colum of phenotype file", type=int, default=0)

    # output
    group_output = parser.add_argument_group('output', 'arguments to specify location of output files')
    group_output.add_argument('--output_prefix', help='prefix output files (suffixes _plot.pdf, _report.txt, _performance.csv)', type=str, default=None)

    #clusterizer
    parser.add_argument('--runner', help=' how to run, for example, "Local()", "Hadoop(...)"', type=str, default='Local()')


    # parse arguments
    args = parser.parse_args()

    # set up logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.getLogger().setLevel(numeric_level)


    # create the runner
    exec("runner = " + args.runner)

    # LL only supported by lmm_full_cv
    if args.strategy != "lmm_full_cv" and args.strategy != "insample_cv" and args.select_by_ll == True:
        logging.critical("Log-likelihood computation only supported for strategy: 'lmm_full_cv' and 'insample_cv' (strategy '%s' was used)" % (args.strategy))
        return

    # covariates not supported for Ridge Regression-based implementations yet
    if args.strategy in ["full_cv", "loo_cv"] and args.cov_fn != None:
        logging.critical("Covariates only supported for the following strategies (strategy '%s' was used): {'lmm_full_cv', 'insample_cv'}" % (args.strategy))
        return

    #standardizer
    #from pysnptools.standardizer import factory
    standardizer = factory(args.standardizer)

    ##############################
    # set up grid
    ##############################
    k_values = [int(x) if x != 'all' else 2147483647 for x in args.k_values.lstrip('[').rstrip(']').lower().split(',')] #old maxint
    delta_values = np.array([np.exp(float(x)) for x in args.ln_delta_values.lstrip('[').rstrip(']').split(',')])

    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    logging.info("grid over k: {0}".format(k_values))
    logging.info("grid over delta: {0}".format(delta_values))


    ##############################
    # invoke model selection
    ##############################

    if os.path.exists(args.snpreader + ".bed") :
        snpreader = Bed(args.snpreader,count_A1=self.count_A1)
    else:
        logging.info("'{0}' + '.bed' doesn't exisit as a file so will evaluate it as an expression".format(args.snpreader))
        eval("snpreader = " + args.snpreader)
        

    fss = FeatureSelectionStrategy(snpreader, args.pheno_fn, args.num_splits, cov_fn=args.cov_fn,  num_pcs=args.num_pcs, test_size=args.test_size, interpolate_delta=args.interpolate_delta, standardizer=standardizer)
    best_k, best_delta, best_obj, best_snps = fss.perform_selection(k_values, delta_values, args.strategy, output_prefix=args.output_prefix, select_by_ll=args.select_by_ll, runner=runner)



    return best_k, best_delta, best_obj, best_snps, fss

class GClass(object):
    @staticmethod
    def factory(snpreader, num_snps_in_memory, standardizer, blocksize,count_A1=None):
        if isinstance(snpreader, str):
            snpreader = Bed(snpreader,count_A1=count_A1)

        if num_snps_in_memory >= snpreader.sid_count:
            in_memory = InMemory(snpreader.read(order='C').standardize(standardizer), standardizer, blocksize)
            in_memory._snpreader.val.flags.writeable = False
            in_memory._val = in_memory._snpreader.val
            return in_memory
        else:
            return FromDisk(snpreader, num_snps_in_memory, standardizer, blocksize, None)

    @property
    def val(self):
        raise NotImplementedError

    @property
    def iid(self):
        raise NotImplementedError

    def kernel(self):
        raise NotImplementedError

    def lin_reg(self, y_train, X_train):
        raise NotImplementedError

class InMemory(GClass):

    def __init__(self, snpreader, standardizer, blocksize):
        self._snpreader = snpreader
        self._standardizer = standardizer
        self._blocksize = blocksize
        self._val = None

    def __repr__(self):
        s = "InMemory({0},{1})".format(self._snpreader,self._standardizer)
        return s


    @property
    def val(self):
        if self._val is None:
            self._snpreader = self._snpreader.read(order='C') #!!LATER when should this be order='F' and when order='C'?
            self._val = self._snpreader.val
            self._val.flags.writeable = False
        return self._val

    @property
    def iid(self):
        return self._snpreader.iid

    def kernel(self):
        self.val # cache the data
        from pysnptools.standardizer import Identity
        return self._snpreader.kernel(Identity())

    def __getitem__(self, iid_indexer_and_snp_indexer):
        iid_indexer, snp_indexer = iid_indexer_and_snp_indexer
        return InMemory(self._snpreader[iid_indexer,snp_indexer], self._standardizer, self._blocksize)

    def lin_reg(self, y_train, X_train):
        return lin_reg.f_regression_block(lin_reg.f_regression_cov_alt, self.val, y_train, blocksize=self._blocksize, C=X_train)




class FromDisk(GClass):

    def __repr__(self):
        s = "FromDisk({0},{1},{2},{3})".format(self._snpreader, self._num_snps_in_memory, self._standardizer, self._indarr_or_none)
        return s

    def __init__(self, snpreader, num_snps_in_memory, standardizer, blocksize, indarr_or_none):
        self._snpreader = snpreader
        self._num_snps_in_memory = num_snps_in_memory
        self._standardizer = standardizer
        self._blocksize = blocksize
        self._indarr_or_none = indarr_or_none
        self.__is_all_slice = sr.SnpReader._is_all_slice(indarr_or_none)
        if self.__is_all_slice:
            self._iid = snpreader.iid
        else:
            self._iid = snpreader.iid[indarr_or_none]

    def kernel(self):
        if self.__is_all_slice:
            K = self._snpreader.kernel(self._standardizer,self._blocksize)
        else:
            K = self._snpreader[self._indarr_or_none,:].kernel(self._standardizer,self._blocksize)
        return K

    @property
    def iid(self):
        return self._iid

    def __getitem__(self, iid_indexer_and_snp_indexer):
        iid_indexer, snp_indexer = iid_indexer_and_snp_indexer

        # use compose_indexer_with_index_or_none instead? (from pysnptools\pysnptools\snpreader\_subset.py)
        if sr.SnpReader._is_all_slice(iid_indexer):
            iid_indexer = self._indarr_or_none
        elif not self.__is_all_slice:
            iid_indexer = self._indarr_or_none[iid_indexer]
        
        if sr.SnpReader._is_all_slice(snp_indexer):
            return FromDisk(self._snpreader, self._num_snps_in_memory, self._standardizer, self._blocksize, iid_indexer)
        else:
            snpreader = self._snpreader[:,snp_indexer].read().standardize(self._standardizer)[iid_indexer,:]
            in_memory = InMemory(snpreader, "No more standardization is expected", self._blocksize)
            return in_memory

    def lin_reg(self, y_train, X_train):
        return f_regression_block_load(lin_reg.f_regression_cov_alt, self._snpreader, self._standardizer, y_train, ind_idx=self._indarr_or_none, blocksize=self._blocksize, C=X_train)

if __name__ == "__main__":
    result = main()


