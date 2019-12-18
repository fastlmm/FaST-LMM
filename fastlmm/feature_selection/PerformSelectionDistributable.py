# std modules
from __future__ import absolute_import
from collections import defaultdict
import gzip
import bz2
import six.moves.cPickle
import time
import os 
import gc
import logging
import io

# common modules
from six.moves import range
import scipy as sp
import numpy as np
import pandas as pd
import sys

# sklearn
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit
from sklearn.datasets import load_boston, load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn import model_selection
from sklearn.decomposition import PCA

# project
import fastlmm.pyplink.plink as plink
import pysnptools.util as pstutil
import fastlmm.util.util as util 
import fastlmm.util.preprocess as up
import fastlmm.inference as fastlmm
import fastlmm.inference.linear_regression as lin_reg
from fastlmm.pyplink.snpset import *  

from pysnptools.snpreader import Bed

class PerformSelectionDistributable(object) : #implements IDistributable

    def __init__(self, feature_selection_strategy, k_values, delta_values, strategy, output_prefix=None, select_by_ll=False,penalty=0.0,create_pdf=True):
        self.feature_selection_strategy = feature_selection_strategy
        self.k_values = k_values
        self.delta_values = delta_values
        self.strategy = strategy
        self.output_prefix = output_prefix
        self.select_by_ll = select_by_ll
        self.penalty=penalty
        self.num_steps_k = len(self.k_values)
        self.num_steps_delta = len(self.delta_values)
        self.create_pdf = create_pdf

    def copyinputs(self, copier):
        copier.input(self.feature_selection_strategy)

    def copyoutputs(self,copier):
        if self.output_prefix != None:
            if self.feature_selection_strategy.num_pcs > 0:
                copier.output(self.feature_selection_strategy.pcs_filename(self.output_prefix))
            copier.output(self.output_prefix + "_snp.csv")
            copier.output(self.output_prefix + "_report.txt")
            if (self.strategy=="lmm_full_cv") or (self.strategy=="insample_cv"):
                for label in ['mse', 'll']:
                    out_fn = self.output_prefix + "_" + label  + ".csv"
                    copier.output(out_fn)
                    if self.create_pdf:
                        xval_fn = self.output_prefix + "_xval_%s.pdf" % label
                        copier.output(xval_fn)
                #if (self.strategy=="lmm_full_cv") and self.feature_selection_strategy.interpolate_delta:
                #    plot_fn=self.output_prefix+"_parabola.pdf"
                #    copier.output(plot_fn)
            else: #this code branch is not regression tested, because it can't be reached.
                plot_fn = self.output_prefix + "_plot.pdf"
                copier.output(plot_fn)

     #start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        return self.feature_selection_strategy.num_folds + 1 #the +1 is for the linreg on the entire dataset

    def work_sequence(self):

        for fold_idx in range(self.feature_selection_strategy.num_folds):
            yield lambda fold_idx=fold_idx : self.dowork(fold_idx)  # the 'fold_idx=fold_idx is need to get around a strangeness in Python
        yield lambda output_prefix=self.output_prefix : self.feature_selection_strategy.linreg_entire_dataset(output_prefix)

    def reduce(self, result_sequence):
        """

        TODO: finish docstring
        return
        
        """
        #self.feature_selection_strategy.run_once() #Don't need and save time

        ##########################################
        ## perform model selection
        ##########################################

        mse_cv = [] #np.zeros((self.feature_selection_strategy.num_folds, len(self.k_values), len(self.delta_values)))
        ll_cv = [] # np.zeros((self.feature_selection_strategy.num_folds, len(self.k_values), len(self.delta_values)))
        best_delta_for_k = [] # np.zeros((self.feature_selection_strategy.num_folds, len(self.k_values)))
        lingreg_results = None
        for result in result_sequence:
            if len(result) == 2: # the lingreg_results look different than the regular results because they are length two
                if lingreg_results != None : raise Exception("assert")
                lingreg_results = result
                continue
            fold_idx, mse_cv1, ll_cv1, best_delta_for_k_1 = result
            mse_cv.append(mse_cv1)
            ll_cv.append(ll_cv1)
            best_delta_for_k.append(best_delta_for_k_1)
        if lingreg_results == None : raise Exception("assert")
        if (self.strategy=="insample_cv") or (self.strategy=="lmm_full_cv"):
            if len(ll_cv) != len(mse_cv) or len(mse_cv) != self.feature_selection_strategy.num_folds:
                raise Exception("These should be the same. len(ll_cv)={0}, len(mse_cv)={1}, self.feature_selection_strategy.num_folds={2}".format(len(ll_cv), len(mse_cv), self.feature_selection_strategy.num_folds))
        else:
            assert len(best_delta_for_k) == len(mse_cv) == self.feature_selection_strategy.num_folds


        # find best parameters
        if mse_cv is not None:
            best_k_mse, best_delta_mse, best_mse, best_delta_mse_interp, best_mse_interp = self.feature_selection_strategy.reduce_result(mse_cv, self.k_values, self.delta_values, self.strategy, self.output_prefix, best_delta_for_k, label="mse",create_pdf=self.create_pdf)
        if ll_cv is not None:
            best_k_ll, best_delta_ll, best_ll, best_delta_ll_interp, best_ll_interp = self.feature_selection_strategy.reduce_result(ll_cv, self.k_values, self.delta_values, self.strategy, self.output_prefix, best_delta_for_k, label="ll",create_pdf=self.create_pdf)

        if self.select_by_ll:
            best_k, best_delta, best_obj, best_delta_interp, best_obj_interp = best_k_ll, best_delta_ll, best_ll, best_delta_ll_interp, best_ll_interp
        else:
            best_k, best_delta, best_obj, best_delta_interp, best_obj_interp = best_k_mse, best_delta_mse, best_mse, best_delta_mse_interp, best_mse_interp

        
        # perform final scan on whole data set
        best_snps, sorted_pval = self.feature_selection_strategy.final_scan(best_k, lingreg_results)

        # write report file
        if self.output_prefix != None:

            report = "k_grid: " + str([k for k in self.k_values]) + "\n"
            ln_delta_grid = np.array([sp.log(x) for x in self.delta_values])
            report += "ln_delta_grid: " + str(ln_delta_grid.tolist()) + "\n"
            report += "best k=%i\nbest ln_delta=%.1e\nbest objective=%.2f" % (best_k, sp.log(best_delta), best_obj)
            if self.feature_selection_strategy.interpolate_delta and best_delta_interp is not None:
                report += "\nbest ln_delta_interp=%.1e\nbest objective_interp=%.2f" % (sp.log(best_delta_interp), best_obj_interp)
            
            report_fn = self.output_prefix + "_report.txt"
            pstutil.create_directory_if_necessary(report_fn)
            report_file = open(report_fn, "w")
            report_file.write(report)
            report_file.close()
            
            # write out SNPs to keep
            #df = pd.DataFrame({"snp_id": best_snps.index, "snp_rs": best_snps.values, "p_values": sorted_pval})
            df = pd.DataFrame({ "snp_rs": best_snps.values}) #Change snp_rs to sid?
            df.to_csv(self.output_prefix + "_snp.csv", index=False, header=False)

        return best_k, best_delta, best_obj, best_snps

    @property
    def tempdirectory(self):
        if self.output_prefix is None:
            return ".work_directory" 
        else:
            return self.output_prefix + ".work_directory"

    def __str__(self):
        if self.output_prefix == None:
            return self.__class__.__name__
        else:
            return "{1} {0}".format(self.__class__.__name__, self.output_prefix)

     #end of IDistributable interface---------------------------------------


    def __repr__(self):
        fp = io.StringIO() if sys.version_info >= (3,0) else io.BytesIO()
        fp.write("{0}(\n".format(self.__class__.__name__))
        varlist = []
        for f in dir(self):
            if f.startswith("_"): # remove items that start with '_'
                continue
            if type(self.__class__.__dict__.get(f,None)) is property: # remove @properties
                continue
            if callable(getattr(self, f)): # remove methods
                continue
            varlist.append(f)
        for var in varlist[:-1]: #all but last
            fp.write("\t{0}={1},\n".format(var, getattr(self, var).__repr__()))
        var = varlist[-1] # last
        fp.write("\t{0}={1})\n".format(var, getattr(self, var).__repr__()))
        result = fp.getvalue()
        fp.close()
        return result


    def dowork(self, fold_idx):
        self.feature_selection_strategy.run_once()
        for i_k,k in enumerate(self.k_values):
            self.k_values[i_k]=min(self.k_values[i_k],self.feature_selection_strategy.snpreader.sid_count)
        max_k = max([1]+[k for k in self.k_values if k != self.feature_selection_strategy.snpreader.sid_count])

        split_iterator = self.feature_selection_strategy.setup_linear_regression(max_k, start=fold_idx, stop=None)
        fold_data = next(split_iterator)

        tt0 = time.time()

        if self.strategy == "lmm_full_cv":
            mse_cv1 = np.zeros((len(self.k_values), len(self.delta_values)))
            ll_cv1 = np.zeros((len(self.k_values), len(self.delta_values)))
            best_delta_for_k_1 = None
        elif self.strategy=="insample_cv":
            mse_cv1 = np.zeros((len(self.k_values)))
            ll_cv1 = np.zeros((len(self.k_values)))
            best_delta_for_k_1 = np.zeros((len(self.k_values)))
        else:
            raise NotImplementedError("not implemented")
        
        logging.info("reporter:counter:PerformSelectionDistributable,foldcount,1")
        for k_idx, k in enumerate(self.k_values):
            logging.info("processing fold={0}, k={1}".format(fold_idx,k))
            logging.info("reporter:status:processing fold={0}, k={1}".format(fold_idx,k))
            logging.info("reporter:counter:PerformSelectionDistributable,k,1")

            model = fastlmm.getLMM()

            # compute kernel externally
            if k == self.feature_selection_strategy.snpreader.sid_count or k >= self.feature_selection_strategy.num_snps_in_memory:
                if k == self.feature_selection_strategy.snpreader.sid_count:
                    # use precomputed kernel
                    logging.info("using precomputed kernel on all snps")
                    K = self.feature_selection_strategy.K
                else:
                    # build kernel in blocks from snpreader (from file)
                    logging.info("building kernel in blocks")
                    top_k_feat_idx = fold_data["feat_idx"][0:int(k)]
                    subset = self.feature_selection_strategy.snpreader[:,top_k_feat_idx]
                    K = subset.kernel(self.feature_selection_strategy.standardizer,blocksize=self.feature_selection_strategy.blocksize)

                train_idx = fold_data["train_idx"]
                test_idx = fold_data["test_idx"]
 
                K_train_lhs = K[train_idx]
                K_train = K_train_lhs[:,train_idx]
                K_train_test = K_train_lhs[:,test_idx].T
                K_test_test = K[test_idx][:,test_idx]

                model.setK(K_train)
                model.setTestData(Xstar=fold_data["X_test"], K0star=K_train_test)

                #np.testing.assert_array_almost_equal(model.K, K_train, decimal=4)
                #np.testing.assert_array_almost_equal(model.Kstar, K_train_test, decimal=4)

            # use precomputed features as before
            else:
                logging.info("using cached data to build kernel")
                outer_G_train = fold_data["G_train"][:,0:k]
                outer_G_test = fold_data["G_test"][:,0:k]
                model.setG(outer_G_train.val)
                model.setTestData(Xstar=fold_data["X_test"], G0star=outer_G_test.val)
                K_test_test = None


            model.sety(fold_data["y_train"])
            model.setX(fold_data["X_train"])

            if self.strategy == "lmm_full_cv":

                for delta_idx, delta_act in enumerate(self.delta_values):
                    if k:
                        delta = delta_act * k
                    else:
                        delta = delta_act
                    REML = True#TODO: Why is REML False?
                    
                    # predict on test set
                    res = model.nLLeval(delta=delta, REML=REML,penalty=self.penalty)
                    out = model.predictMean(beta=res["beta"], delta=delta)
                    mse_cv1[k_idx, delta_idx] = mean_squared_error(fold_data["y_test"], out)
                    ll_cv1[k_idx, delta_idx] = model.nLLeval_test(fold_data["y_test"], res["beta"], sigma2=res["sigma2"], delta=delta, Kstar_star=K_test_test)

            elif self.strategy == "insample_cv":

                best_res = None
                best_delta = None
                best_nLL = float("inf")
                REML = True

                #Note that for brent = True there will always be many unique delta values, as these deiate from the grid.
                #brent = False
                brent = True

                # evaluate negative log-likelihood for different values of alpha
                import fastlmm.util.mingrid as mingrid
                resmin = [None]
                def f(x):
                    if k:
                        delta_corr = x * k
                    else:
                        delta_corr = x
                    myres = model.nLLeval(delta = delta_corr, REML = REML,penalty=self.penalty)
                    if (resmin[0] is None) or (myres['nLL']<resmin[0]['nLL']):
                        resmin[0]=myres
                        resmin[0]["delta_corr"] = delta_corr
                        resmin[0]["delta"] = x
                    return myres["nLL"]
                res = mingrid.minimize1D(f,evalgrid = self.delta_values,brent = brent)

                if 0:#old code without brent search
                    for delta_idx, delta_act in enumerate(self.delta_values):
                        delta = delta_act * k #rescale delta for X val.
                        res = model.nLLeval(delta=delta,REML=REML,penalty=self.penalty)
                        #TODO: check if we need scale
                    
                        if res["nLL"] < best_nLL:
                            best_res = res
                            best_delta_act = delta_act
                            best_delta = delta
                            best_nLL = res["nLL"]
                out = model.predictMean(beta=resmin[0]["beta"], delta=resmin[0]["delta_corr"])
                mse_cv1[k_idx] = mean_squared_error(fold_data["y_test"], out)
                ll_cv1[k_idx] = model.nLLeval_test(fold_data["y_test"], resmin[0]["beta"], sigma2=resmin[0]["sigma2"], delta=resmin[0]["delta_corr"], Kstar_star=K_test_test)
                best_delta_for_k_1[k_idx] = resmin[0]["delta"]

        logging.info("crossval time %.2f s" % (float(time.time() - tt0)))

        return fold_idx, mse_cv1, ll_cv1, best_delta_for_k_1


def build_kernel_blocked(snpreader, snp_idx=None, blocksize=10000,alt_snpreader=None,allowlowrank=False):
    """build kernel by loading blocks of SNPs
    """
    if alt_snpreader is None:
        alt_snpreader = snpreader
            
    if hasattr(alt_snpreader,"ind_used") and alt_snpreader.ind_used is not None:
        N = len(alt_snpreader.ind_used)
    else:
        N = len(alt_snpreader.original_iids)
    
    t0 = time.time()

    K = sp.zeros([N,N])
    num_snps = alt_snpreader.snp_count

    if snp_idx != None:
        snp_names = alt_snpreader.rs[snp_idx]
        current_size = len(snp_names)
        logging.info("reading %i SNPs in blocks of %i and adding up kernels" % (len(snp_idx), blocksize))
    else:
        current_size = num_snps
        logging.info("constructing K from all %i SNPs (for %i individuals)" % (num_snps, N))

    ct = 0
    ts = time.time()

    if (not allowlowrank) and alt_snpreader.snp_count<N: raise Exception("need to adjust code to handle low rank")

    for start in range(0, current_size, blocksize):
        ct += blocksize

        if snp_idx == None:
            tmp_set = PositionRange(start, blocksize)
        else:
            tmp_set = SnpAndSetName('someset', snp_names[start:start+blocksize])

        snps = alt_snpreader.read(tmp_set)['snps']
        snps = up.standardize(snps)

        #logging.info("start = {0}".format(start))
        K += snps.dot(snps.T)

        if ct % blocksize==0:
            logging.info("read %s SNPs in %.2f seconds" % (ct, time.time()-ts))


    # normalize kernel
    #K = K/sp.sqrt(alt_snpreader.snp_count)

    #K = K + 1e-5*sp.eye(N,N)     
    t1 = time.time()
    logging.info("%.2f seconds elapsed" % (t1-t0))

    return K

