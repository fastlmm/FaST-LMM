from __future__ import absolute_import
def qf(chi2val, coeffs,dof = None,noncentrality = None,sigma = 0.0,lim=1000000,acc=1e-08):
    import numpy as np
    from fastlmm.util.stats.quadform.qfc_src import wrap_qfc
    size = coeffs.shape[0]
    if dof is None:
        dof = np.ones(size,dtype = 'int32')
        #dof = np.ones(size)
    if noncentrality is None:
        noncentrality = np.zeros(size)
    ifault=np.zeros(1,dtype = 'int32')
    #ifault=np.zeros(1)
    trace = np.zeros(7)
    #import pdb
    #pdb.set_trace()
    pval = 1.0-wrap_qfc.qf(coeffs,noncentrality,dof,sigma,chi2val,lim,acc,trace,ifault)
    return pval, ifault[0], trace
    
if __name__ == "__main__":
    mix = np.ones(1)
    c = 1.0
    res = qf(c,mix)
