from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import logging
import unittest
import os
import scipy.linalg as LA
import time
from sklearn.utils import safe_sqr, check_array
from scipy import stats

from pysnptools.snpreader import Bed,Pheno
from pysnptools.snpreader import SnpData,SnpReader
from pysnptools.kernelreader import KernelNpz
from pysnptools.kernelreader import SnpKernel
from pysnptools.kernelreader import KernelReader
from pysnptools.kernelreader import Identity as KernelIdentity
import pysnptools.util as pstutil
from pysnptools.standardizer import DiagKtoN,UnitTrained
from pysnptools.standardizer import Unit
from pysnptools.util import intersect_apply
from pysnptools.standardizer import Standardizer
from fastlmm.inference.lmm import LMM
from fastlmm.inference.fastlmm_predictor import _pheno_fixup
from fastlmm.inference import FastLMM
from pysnptools.standardizer import Identity as StandardizerIdentity
from scipy.stats import multivariate_normal
from fastlmm.util.pickle_io import load, save

# make FastLmm use this when there are no SNPs or K is Identity?
class LinearRegression(object):
    '''
    A linear regression predictor, that works like the FastLMM in fastlmm_predictor.py, but that expects all similarity matrices to be identity. 

    **Constructor:**
        :Parameters: * **covariate_standardizer** (:class:`Standardizer`) -- The PySnpTools standardizer to be apply to X, the covariate data. Some choices include :class:`Standardizer.Unit` (Default. Fills missing with zero) and :class:`Standardizer.Identity` (do nothing)

        :Example:

        >>> import numpy as np
        >>> import logging
        >>> from pysnptools.snpreader import Pheno
        >>> from fastlmm.inference import LinearRegression
        >>> from fastlmm.util import example_file # Download and return local file name
        >>> logging.basicConfig(level=logging.INFO)
        >>> cov = Pheno(example_file("fastlmm/feature_selection/examples/toydata.cov"))
        >>> pheno_fn = example_file("fastlmm/feature_selection/examples/toydata.phe")
        >>> train_idx = np.r_[10:cov.iid_count] # iids 10 and on
        >>> test_idx  = np.r_[0:10] # the first 10 iids
        >>> linreg = LinearRegression()
        >>> #We give it phenotype information for extra examples, but it reorders and intersects the examples, so only training examples are used. 
        >>> _ = linreg.fit(X=cov[train_idx,:],y=pheno_fn) 
        >>> mean, covariance = linreg.predict(X=cov[test_idx,:])
        >>> print(mean.iid[0], round(mean.val[0,0],7), round(covariance.val[0,0],7))
        ['per0' 'per0'] 0.1518764 0.9043703
        >>> nll = linreg.score(X=cov[test_idx,:],y=pheno_fn)
        >>> print(round(nll,7))
        13.6688448


        '''
    def __init__(self,covariate_standardizer=Unit()):
        self.covariate_standardizer = covariate_standardizer
        self.is_fitted = False

    def fit(self, X=None, y=None, K0_train=None, K1_train=None, h2=None, mixing=None,count_A1=None):
        """
        Method for training a :class:`FastLMM` predictor. If the examples in X, y, K0_train, K1_train are not the same, they will be reordered and intersected.

        :param X: training covariate information, optional: 
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type X: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param y: training phenotype:
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type y: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param K0_train: Must be None. Represents the identity similarity matrix.
        :type K0_train: None

        :param K1_train: Must be None. Represents the identity similarity matrix.
        :type K1_train: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param h2: Ignored. Optional.
        :type h2: number

        :param mixing: Ignored. Optional.
        :type mixing: number

        :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
             alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
        :type count_A1: bool


        :rtype: self, the fitted Linear Regression predictor
        """
        self.is_fitted = True
        assert K0_train is None # could also accept that ID or no snps
        assert K1_train is None # could also accept that ID or no snps

        assert y is not None, "y must be given"

        y = _pheno_fixup(y,count_A1=count_A1)
        assert y.sid_count == 1, "Expect y to be just one variable"
        X = _pheno_fixup(X, iid_if_none=y.iid,count_A1=count_A1)

        X, y  = intersect_apply([X, y])
        y = y.read()
        X, covar_unit_trained = X.read().standardize(self.covariate_standardizer,return_trained=True)

        # add a column of 1's to cov to increase DOF of model (and accuracy) by allowing a constant offset
        X = SnpData(iid=X.iid,
                                sid=FastLMM._new_snp_name(X),
                                val=np.c_[X.val,np.ones((X.iid_count,1))])


        lsqSol = np.linalg.lstsq(X.val, y.val[:,0],rcond=-1)
        bs=lsqSol[0] #weights
        r2=lsqSol[1] #squared residuals
        D=lsqSol[2]  #rank of design matrix
        N=y.iid_count

        self.beta = bs
        self.ssres = float(r2)
        self.sstot = ((y.val-y.val.mean())**2).sum()
        self.covar_unit_trained = covar_unit_trained
        self.iid_count = X.iid_count
        self.covar_sid = X.sid
        self.pheno_sid = y.sid
        return self
    

    def predict(self,X=None,K0_whole_test=None,K1_whole_test=None,iid_if_none=None,count_A1=None):
        """
        Method for predicting from a fitted :class:`FastLMM` predictor.
        If the examples in X, K0_whole_test, K1_whole_test are not the same, they will be reordered and intersected.

        :param X: testing covariate information, optional: 
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type X: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param K0_whole_test: Must be None. Represents the identity similarity matrix.
        :type K0_whole_test: None

        :param K1_whole_test: Must be None. Represents the identity similarity matrix.
        :type K1_whole_test: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param iid_if_none: Examples to predict for if no X, K0_whole_test, K1_whole_test is provided.
        :type iid_if_none: an ndarray of two strings

        :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
             alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
        :type count_A1: bool

        :rtype: A `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__ of the means and a :class:`KernelData` of the covariance
        """

        assert self.is_fitted, "Can only predict after predictor has been fitted"
        assert K0_whole_test is None or isinstance(K0_whole_test,KernelIdentity) # could also accept no snps
        assert K1_whole_test is None or isinstance(K1_whole_test,KernelIdentity) # could also accept no snps

        X = _pheno_fixup(X,iid_if_none=iid_if_none,count_A1=count_A1)
        X = X.read().standardize(self.covar_unit_trained)

        # add a column of 1's to cov to increase DOF of model (and accuracy) by allowing a constant offset
        X = SnpData(iid=X.iid,
                              sid=FastLMM._new_snp_name(X),
                              val=np.c_[X.read().val,np.ones((X.iid_count,1))])
        assert np.array_equal(X.sid,self.covar_sid), "Expect covar sids to be the same in train and test."

        pheno_predicted = X.val.dot(self.beta).reshape(-1,1)
        ret0 = SnpData(iid = X.iid, sid=self.pheno_sid,val=pheno_predicted,pos=np.array([[np.nan,np.nan,np.nan]]),name="linear regression Prediction") #!!!replace 'parent_string' with 'name'

        from pysnptools.kernelreader import KernelData
        ret1 = KernelData(iid=X.iid,val=np.eye(X.iid_count)* self.ssres / self.iid_count)
        return ret0, ret1

    def score(self, X=None, y=None, K0_whole_test=None, K1_whole_test=None, iid_if_none=None, return_mse_too=False, count_A1=None):
        """
        Method for calculating the negative log likelihood of testing examples.
        If the examples in X,y,  K0_whole_test, K1_whole_test are not the same, they will be reordered and intersected.

        :param X: testing covariate information, optional: 
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type X: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param y: testing phenotype:
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type y: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param K0_whole_test: Must be None. Represents the identity similarity matrix.
        :type K0_whole_test: None

        :param K1_whole_test: Must be None. Represents the identity similarity matrix.
        :type K1_whole_test: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param iid_if_none: Examples to predict for if no X, K0_whole_test, K1_whole_test is provided.
        :type iid_if_none: an ndarray of two strings

        :param return_mse_too: If true, will also return the mean squared error.
        :type return_mse_too: bool

        :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
             alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
        :type count_A1: bool

        :rtype: a float of the negative log likelihood and, optionally, a float of the mean squared error.
        """
        mean0, covar0 = self.predict(K0_whole_test=K0_whole_test,K1_whole_test=K1_whole_test,X=X,iid_if_none=iid_if_none,count_A1=count_A1)
        y = _pheno_fixup(y, iid_if_none=covar0.iid,count_A1=count_A1)
        mean, covar, y = intersect_apply([mean0, covar0, y])
        var = multivariate_normal(mean=mean.read(order='A',view_ok=True).val.reshape(-1), cov=covar.read(order='A',view_ok=True).val)
        y_actual = y.read().val
        nll = -np.log(var.pdf(y_actual.reshape(-1)))
        if not return_mse_too:
            return nll
        else:
            mse = ((y_actual-mean)**2).sum()
            return nll, mse





"""
Created on 2013-08-02
@author: Christian Widmer <chris@shogun-toolbox.org>
@summary: Module for univariate feature selection in the presence of covariates


Motivated by sklearn's linear regression method for feature
selection, we've come up with an extended version that takes
care of covariates

based on sklearn code (f_regression):
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_selection/univariate_selection.py

"""




#def get_example_data():
#    """
#    load plink files
#    """

#    import fastlmm.pyplink.plink as plink
#    import pysnptools.snpreader.bed as Bed
#    import fastlmm.util.util as util


#    ipheno = 0
#    foldIter = 0


#    """
#    import dataset
#    dat = dataset.importDataset("pheno4")

#    fn_bed = dat["bedFile"]
#    fn_pheno = dat["phenoFile"]
#    """

#    fn_bed = "../featureSelection/examples/toydata.5chrom.bed"
#    fn_pheno = "../feature_selection/examples/toydata.phe"

#    import pysnptools.util.pheno as pstpheno
#    pheno = pstpheno.loadPhen(fn_pheno)

#    # load data
#    bed = plink.Bed(fn_bed)

#    indarr = util.intersect_ids([pheno['iid'],bed.iid])

#    pheno['iid'] = pheno['iid'][indarr[:,0]]
#    pheno['vals'] = pheno['vals'][indarr[:,0]]
#    bed = bed[indarr[:,1],:]

#    N = pheno['vals'].shape[0]
#    y = pheno['vals'][:,ipheno]
#    iid = pheno['iid']

#    snps = bed.read().standardize()

#    return snps, y


def f_regression_block(fun,X,y,blocksize=None,**args):
   """
   runs f_regression for each block separately (saves memory).

   -------------------------
   fun  : method that returns statistics,pval
   X    : {array-like, sparse matrix}  shape = (n_samples, n_features)
          The set of regressors that will tested sequentially.
   y    : array of shape(n_samples).
          The data matrix
   blocksize    : number of SNPs per block
   """
   if blocksize==None:
       return fun(X,y,**args)

   idx_start = 0
   idx_stop = int(blocksize)

   pval = np.zeros(X.shape[1])
   stats = np.zeros(X.shape[1])

   while idx_start<X.shape[1]:
        stats[idx_start:idx_stop], pval[idx_start:idx_stop] = fun(X[:,idx_start:idx_stop],y,**args)

        idx_start = idx_stop
        idx_stop += blocksize
        if idx_stop>X.shape[1]:
            idx_stop = X.shape[1]

   return stats,pval


def f_regression_cov_alt(X, y, C):
    """
    Implementation as derived in tex document

    See pg 12 of following document for definition of F-statistic
    http://www-stat.stanford.edu/~jtaylo/courses/stats191/notes/simple_diagnostics.pdf

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will tested sequentially.

    y : array of shape(n_samples).
        The data matrix

    c : {array-like, sparse matrix}  shape = (n_samples, n_covariates)
        The set of covariates.


    Returns
    -------
    F : array, shape=(n_features,)
        F values of features.

    pval : array, shape=(n_features,)
        p-values of F-scores.
    """
    # make sure we don't overwrite input data
    old_flag_X = X.flags.writeable
    old_flag_C = C.flags.writeable
    old_flag_y = y.flags.writeable
    X.flags.writeable = False
    C.flags.writeable = False
    y.flags.writeable = False


    #X, C, y = check_array(X, C, y, dtype=np.float)
    y = y.ravel()

    # make copy of input data
    X = X.copy(order="F")
    y = y.copy()

    assert C.shape[1] < C.shape[0]
    cpinv = np.linalg.pinv(C)
    X -= np.dot(C,(np.dot(cpinv, X))) #most expensive line (runtime)
    y -= np.dot(C,(np.dot(cpinv, y)))

    yS = safe_sqr(y.T.dot(X)) # will create a copy

    # Note: (X*X).sum(0) = X.T.dot(X).diagonal(), computed efficiently
    # see e.g.: http://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-dot-product-calculating-only-the-diagonal-entries-of-the
    # TODO: make this smarter using either stride tricks or cython
    X *= X
    denom = X.sum(0) * y.T.dot(y) - yS
    F = yS / denom

    # degrees of freedom
    dof = (X.shape[0] - 1 - C.shape[1]) / (1) #(df_fm / (df_rm - df_fm))
    F *= dof

    # convert to p-values
    pv = stats.f.sf(F, 1, dof)

    # restore old state
    X.flags.writeable = old_flag_X
    C.flags.writeable = old_flag_C
    y.flags.writeable = old_flag_y

    return F, pv


def f_regression_cov(X, y, C):
    """Univariate linear regression tests

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.

    This is done in 3 steps:
    1. the regressor of interest and the data are orthogonalized
    wrt constant regressors
    2. the cross correlation between data and regressors is computed
    3. it is converted to an F score then to a p-value

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will tested sequentially.

    y : array of shape(n_samples).
        The data matrix

    c : {array-like, sparse matrix}  shape = (n_samples, n_covariates)
        The set of covariates.


    Returns
    -------
    F : array, shape=(n_features,)
        F values of features.

    pval : array, shape=(n_features,)
        p-values of F-scores.
    """

    X = check_array(X, dtype=np.float)
    C = check_array(C, dtype=np.float)
    y = check_array(y, dtype=np.float)    
    y = y.ravel()

    assert C.shape[1] < C.shape[0]
    cpinv = np.linalg.pinv(C)
    X -= np.dot(C,(np.dot(cpinv, X)))
    y -= np.dot(C,(np.dot(cpinv, y)))

    # compute the correlation
    corr = np.dot(y, X)
    corr /= np.asarray(np.sqrt(safe_sqr(X).sum(axis=0))).ravel()
    corr /= np.asarray(np.sqrt(safe_sqr(y).sum())).ravel()

    # convert to p-value
    dof = (X.shape[0] - 1 - C.shape[1]) / (1) #(df_fm / (df_rm - df_fm))
    F = corr ** 2 / (1 - corr ** 2) * dof
    pv = stats.f.sf(F, 1, dof)
    return F, pv


def test_bias():
    """
    make sure we get the same result for setting C=unitvec
    """

    S, y = get_example_data()
    C = np.ones((len(y),1))

    from sklearn.feature_selection import f_regression

    F1, pval1 = f_regression(S, y, center=True)
    F2, pval2 = f_regression_cov(S, C, y)
    F3, pval3 = f_regression_cov_alt(S, C, y)

    # make sure values are the same
    np.testing.assert_array_almost_equal(F1, F2)
    np.testing.assert_array_almost_equal(F2, F3)
    np.testing.assert_array_almost_equal(pval1, pval2)
    np.testing.assert_array_almost_equal(pval2, pval3)


def test_cov():
    """
    compare different implementations, make sure results are the same
    """

    S, y = get_example_data()
    C = S[:,0:10]
    S = S[:,10:]

    F1, pval1 = f_regression_cov(S, C, y)
    F2, pval2 = f_regression_cov_alt(S, C, y)

    np.testing.assert_array_almost_equal(F1, F2)
    np.testing.assert_array_almost_equal(pval1, pval2)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
    #test_cov()
    #test_bias()
