from __future__ import absolute_import
from __future__ import print_function
import time
import os
import sys

import scipy as SP
import sklearn.metrics as SKM

import sklearn.feature_selection as SKFS
import sklearn.model_selection as SKCV 
import sklearn.metrics as SKM

from pysnptools.util.mapreduce1.distributable import *
from pysnptools.util.mapreduce1.runner import *
import fastlmm.pyplink.plink as plink
import fastlmm.pyplink.Bed as Bed
import fastlmm.pyplink.snpset.PositionRange as PositionRange
import fastlmm.pyplink.snpset.SnpSetAndName as SnpSetAndName
import fastlmm.util.util as util
import fastlmm.inference as fastlmm

from .feature_selection_cv import load_snp_data
from six.moves import range

class KernelRidgeCV(): # implements IDistributable
    '''
    A class for running cross-validation on kernel ridge regression: The method determines the best regularization parameter alpha by
    cross-validating it.

    The Rdige regression optimization problem is given by:
    min  1./2n*||y - Xw||_2^2 + alpha * ||w||_2
    '''

    def __init__(self, bed_fn, pheno_fn, num_folds,random_state=None,cov_fn=None,offset=True):
        """set up kernel ridge regression
        ----------

        bed_fn : str
            File name of binary SNP file

        pheno_fn : str
            File name of phenotype file

        num_folds : int
            Number of folds in k-fold cross-validation

        cov_fn : str, optional, default=None
            File name of covariates file

        offset : bool, default=True
            adds offset to the covariates specified in cov_fn, if necessary
        """
 
        # data file names
        self.bed_fn = bed_fn
        self.pheno_fn = pheno_fn
        self.cov_fn = cov_fn

        # optional parameters
        self.num_folds = num_folds
        self.fold_to_train_data = None
        self.random_state = random_state
        self.offset = offset
        self.K = None

    def perform_selection(self,delta_values,strategy,plots_fn=None,results_fn=None):
        """Perform delta selection for kernel ridge regression

        delta_values : array-like, shape = [n_steps_delta]
            Array of delta values to test

        strategy : {'full_cv','insample_cv'}
            Strategy to perform feature selection:
            - 'full_cv' perform cross-validation over delta
            - 'insample_cv' pestimates delta in sample using maximum likelihood.

        plots_fn    : str, optional, default=None
            File name for generated plot. if not specified, the plot is not saved

        results_fn  : str, optional, default=None
            file name for saving cross-validation results. if not specified, nothing is saved
        Returns
        -------
        best_delta : float
            best regularization parameter delta for ridge regression

        """
        import matplotlib.pylab as PLT 


        # use precomputed data if available
        if self.K == None:
            self.setup_kernel()

        print('run selection strategy %s'%strategy)

        model = fastlmm.lmm()
        nInds = self.K.shape[0]
   
        if strategy=='insample':
            # take delta with largest likelihood
            model.setK(self.K)
            model.sety(self.y)
            model.setX(self.X)
            best_delta = None
            best_nLL = SP.inf

            # evaluate negative log-likelihood for different values of alpha
            nLLs = SP.zeros(len(delta_values))
            for delta_idx, delta in enumerate(delta_values):
                res = model.nLLeval(delta=delta,REML=True)
                if res["nLL"] < best_nLL:
                    best_delta = delta
                    best_nLL = res["nLL"]

                nLLs[delta_idx] = res['nLL']

            fig = PLT.figure()
            fig.add_subplot(111)
            PLT.semilogx(delta_values,nLLs,color='g',linestyle='-')
            PLT.axvline(best_delta,color='r',linestyle='--')
            PLT.xlabel('logdelta')
            PLT.ylabel('nLL')
            PLT.title('Best delta: %f'%best_delta)
            PLT.grid(True)
            if plots_fn!=None:
                PLT.savefig(plots_fn)
            if results_fn!=None:
                SP.savetxt(results_fn, SP.vstack((delta_values,nLLs)).T,delimiter='\t',header='delta\tnLLs')
            
        if strategy=='cv':
            # run cross-validation for determining best delta
            kfoldIter = SKCV.KFold(n_splits=self.num_folds,shuffle=True,random_state=self.random_state).split(list(range(nInds)))
            Ypred = SP.zeros((len(delta_values),nInds))
            for Itrain,Itest in kfoldIter:
                model.setK(self.K[Itrain][:,Itrain])
                model.sety(self.y[Itrain])
                model.setX(self.X[Itrain])

                model.setTestData(Xstar=self.X[Itest],K0star=self.K[Itest][:,Itrain])
                
                for delta_idx,delta in enumerate(delta_values):
                    res = model.nLLeval(delta=delta,REML=True)
                    beta = res['beta']
                    Ypred[delta_idx,Itest] = model.predictMean(beta=beta,delta=delta)

            MSE = SP.zeros(len(delta_values))
            for i in range(len(delta_values)):
                MSE[i] = SKM.mean_squared_error(self.y,Ypred[i])
            idx_bestdelta = SP.argmin(MSE)
            best_delta = delta_values[idx_bestdelta]

            fig = PLT.figure()
            fig.add_subplot(111)
            PLT.semilogx(delta_values,MSE,color='g',linestyle='-')
            PLT.axvline(best_delta,color='r',linestyle='--')
            PLT.xlabel('logdelta')
            PLT.ylabel('MSE')
            PLT.grid(True)
            PLT.title('Best delta: %f'%best_delta)
            if plots_fn!=None:
                PLT.savefig(plots_fn)
            if results_fn!=None:
                SP.savetxt(results_fn, SP.vstack((delta_values,MSE)).T,delimiter='\t',header='delta\tnLLs')

        return best_delta
    

    
    def setup_kernel(self):
        """precomputes the kernel
        """
        print("loading data...")
        G, self.X, self.y = load_snp_data(self.bed_fn, self.pheno_fn, cov_fn=self.cov_fn,offset=self.offset)
        print("done.")
        print("precomputing kernel... ")
        nSnps = G.shape[1]
        self.K = 1./nSnps * SP.dot(G,G.T)
        print("done.")
        del G
   

 