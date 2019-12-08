from __future__ import absolute_import
from .Beta import *
from .Unit import *

#import warnings
#warnings.warn("This __init__.py is deprecated. Pysnptools includes newer version", DeprecationWarning)

def factory(s):
    s = s.capitalize()
    if s == "Unit" or s=="Unit()":
        return Unit()

    if s=="Beta":
        return Beta()

    if s.startswith("Beta("):
        standardizer = eval(s)
        return standardizer

def standardize_with_lambda(snps, lambdax, blocksize = None):
    if blocksize==None:
       return lambdax(snps)

    idx_start = 0
    idx_stop = blocksize

    while idx_start<snps.shape[1]:
        #print idx_start
        lambdax(snps[:,idx_start:idx_stop])

        idx_start = idx_stop
        idx_stop += blocksize
        if idx_stop>snps.shape[1]:
            idx_stop = snps.shape[1]

    return snps

def standardize_unit_python(snps, returnStats=False):
    '''
    standardize snps to zero-mean and unit variance
    '''

    N = snps.shape[0]
    S = snps.shape[1]

    imissX = np.isnan(snps)
    snp_sum =  np.nansum(snps,axis=0)
    n_obs_sum = (~imissX).sum(0)
    
    snp_mean = (snp_sum*1.0)/n_obs_sum
    snps -= snp_mean
    snp_std = np.sqrt(np.nansum(snps**2, axis=0)/n_obs_sum)

    # avoid div by 0 when standardizing
    if snp_std.__contains__(0.0):
        logging.warning("A least one snps has only one value, that is, its standard deviation is zero")
    snp_std[snp_std == 0.0] = 1.0
    snps /= snp_std
    snps[imissX] = 0
    
    if returnStats:
        return snps,snp_mean,snp_std

    return snps

def standardize_beta_python(snps, betaA, betaB):
    '''
    standardize snps with Beta prior
    '''

    N = snps.shape[0]
    S = snps.shape[1]

    imissX = np.isnan(snps)
    snp_sum =  np.nansum(snps,axis=0)
    n_obs_sum = (~imissX).sum(0)
    
    snp_mean = (snp_sum*1.0)/n_obs_sum
    snps -= snp_mean
    snp_std = np.sqrt(np.nansum(snps**2, axis=0)/n_obs_sum)
    if snp_std.__contains__(0.0):
        logging.warning("A least one snps has only one value, that is, its standard deviation is zero")

    maf = snp_mean/2.0
    maf[maf>0.5]=1.0- maf[maf>0.5]

    # avoid div by 0 when standardizing
    import scipy.stats as st
    maf_beta = st.beta.pdf(maf, betaA, betaB)
    snps*=maf_beta
    snps[imissX] = 0.0
     
    return snps
