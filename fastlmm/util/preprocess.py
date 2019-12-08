from __future__ import absolute_import
import numpy as np
import time
import fastlmm.util.standardizer as stdizer
import warnings

#TODO: wrap C-stuff here?
def mean_impute(X, imissX=None, maxval=2.0):
    '''
    mean impute an array of floats
    ---------------------------------------------------
    X           data array
    imissX      indicator array of missing values
    maxval      larges value in X (default 2.0)
    ---------------------------------------------------
    '''
    if imissX is None:
        imissX = np.isnan(X)

    n_i,n_s=X.shape
    if imissX is None:
        n_obs_SNP=np.ones(X.shape)
    else:
        i_nonan=(~imissX)
        n_obs_SNP=i_nonan.sum(0)
        X[imissX]=0.0
    snp_sum=(X).sum(0)
    one_over_sqrt_pi=(1.0+snp_sum)/(2.0+maxval*n_obs_SNP)
    one_over_sqrt_pi=1./np.sqrt(one_over_sqrt_pi*(1.-one_over_sqrt_pi))
    snp_mean=(snp_sum*1.0)/(n_obs_SNP)

    X_ret=X-snp_mean
    X_ret*=one_over_sqrt_pi
    if imissX is not None:
        X_ret[imissX]=0.0
    return X_ret


def standardize(snps, blocksize=None, standardizer=stdizer.Unit(), force_python_only=False):
    '''Does in-place standardization.
            Will use C++ if possible (for single and double, unit and beta, order="F" and order="C")
    '''
    #!!warnings.warn("This standardizer is deprecated. Pysnptools includes newer versions of standardization", DeprecationWarning)
    if isinstance(standardizer, str):
        standardizer = standardizer.factor(standardizer)

    if blocksize is not None and blocksize >= snps.shape[1]: #If blocksize is larger than the # of snps, set it to None
        blocksize = None

    return standardizer.standardize(snps, blocksize=blocksize, force_python_only = force_python_only)