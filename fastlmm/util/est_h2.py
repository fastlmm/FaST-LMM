from __future__ import absolute_import
from __future__ import print_function
import fastlmm.inference.lmm_cov as lmm_cov
import numpy as np

def est_h2(Y, K, covariates=None, nGridH2=10000, plot=True, verbose=True):
    """
    This function implements the Bayesian heritability estimate from Furlotte et al., 2014

    Furlotte, Nicholas A., David Heckerman, and Christoph Lippert.
    "Quantifying the uncertainty in heritability." Journal of human genetics 59.5 (2014): 269-275.

    Args:
        Y:              [N x 1] np.ndarray of phenotype values
        K:              [N x N] np.ndarray of kinship values
        covariates:     [N x D] np.ndarray of covariate values [default: None]
        plot:           Boolean, create a plot? [default: True]
        verbose:        print results? [default: True]

    returns:
        REML estimate of h^2 (as in Yang et al. 2010)
        posterior mean of h^2
        posterior variance of h^2
        h2 values on a grid
        posterior for h2 values on a grid
    """
    lmm = lmm_cov.LMM(forcefullrank=False, X=covariates, linreg=None, Y=Y, G=None, K=K, regressX=True, inplace=False)
    h2 = lmm.findH2()
    h2_posterior = lmm.posterior_h2(nGridH2=nGridH2)
    logp = -h2_posterior[2]
    grid = h2_posterior[1]
    post_h2 = np.exp(logp-logp.max())/np.exp(logp-logp.max()).sum()*logp.shape[0]
    h2_mean = (np.exp(logp-logp.max())*grid[:,np.newaxis]).sum()/np.exp(logp-logp.max()).sum()
    h2_var = (np.exp(logp-logp.max())*(grid[:,np.newaxis] - h2_mean)**2.0).sum()/np.exp(logp-logp.max()).sum()
    if plot:
        import pylab as plt
        plt.figure()
        plt.plot([h2['h2'],h2['h2']],[0,1],"r")
        plt.plot([h2_mean,h2_mean],[0,1],"g")
        plt.legend(["REML-estimate = %.3f"%h2['h2'],"posterior mean = %.3f"%h2_mean])
        plt.plot(grid.flatten(),post_h2.flatten())
        plt.xlabel("$h^2$")
        plt.ylabel("$p( h^2 | Data)$")
    if verbose:
        print("max[h^2] = %.5f  E[h^2] = %.5f +- %.5f" % (h2['h2'], h2_mean,np.sqrt(h2_var)))
    return h2, h2_mean, h2_var, grid, post_h2 