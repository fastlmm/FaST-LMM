from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.stats as st
import scipy.interpolate
import scipy.linalg as la
import fastlmm.util.preprocess as util
import scipy.stats as st
import time
from six.moves import range
#import ipdb

def print_missing_info(y):
    print(str(np.isnan(y).sum().sum()) + " missing items out of " + str(np.product(y.shape)) + "=" + str(100.0*np.isnan(y).sum().sum()/np.product(y.shape)) + "%")


#would be nice to make a low-mem version of this, but this requires
#2-passes, because we must mean-center not only the SNPs, but also the
#individuals
def TESTBEFOREUSING_pca_covariates(bedfile,npc,outfile=None):
    '''
    Read in bed file and compute PC covariates (a la Eigenstrat)
    bedfile should include the .bed extension
    returns pccov, U, S
    '''    
    import fastlmm.pyplink.plink as pl
    import os.path
    import time
    import scipy.linalg as la
    import numpy as np
    root, ext = os.path.splitext(bedfile)
    print("reading in bed file...")

    t0=time.time()   
    SNPs = Bed(root).read().standardize()
    t1=time.time()   
    print(("Elapsed time %.2f seconds" % (t1-t0)))

    [N,M]=snps.shape
    print("found nind=" + str(N) + ", nsnp=" + str(M))
    if M<N: 
        print("only doing full rank, should use low rank here")
    
    #not needed, and in practice, makes no difference
    #mean center the individuals    
    #meanval=sp.mean(snps,axis=0)
    #snps=snps-meanval
    
    print("computing kernel...")
    t0=time.time()           
    K=sp.dot(snps,snps.T)  
    t1=time.time()       
    print(("Elapsed time %.2f seconds" % (t1-t0)))  
    snps=None
    
    t0=time.time()   
    print("computing svd...")
    [U,S,V] = la.svd(K,full_matrices = False) #!!!use big_svd?
    t1=time.time()   
    print(("Elapsed time %.2f seconds" % (t1-t0)))  
    
    S=sp.sqrt(S)
    #UShalf=sp.dot(U,sp.diag(S)) #expensive
    UShalf=sp.multiply(U,np.tile(S,(N,1))) #faster, but not as fast as as_strided which I can't get to work
    pccov=UShalf[:,0:npc]
    print("done.")
    if outfile is not None:
        import fastlmm.util.util as ut        
        ut.write_plink_covariates(SNPs['iid'],pccov,outfile)
    return pccov,U,S

def lrtpvals_qqfit_file(filein, qmax=0.1):

    import fastlmm.util.stats.chi2mixture as c2
    import fastlmm.util.stats.plotp
    import pandas as pd          
    colname="2*(LL(alt)-LL(null))"
    lrtperm=pd.read_csv(filein,delimiter = '\t',dtype={colname:np.float64},usecols=[colname])[colname].values    
    print("found " + str(len(lrtperm)) + "null test stats")

    mix = c2.chi2mixture( lrt = lrtperm, qmax = qmax, alteqnull = None)     
    res = mix.fit_params_Qreg() # paramter fitting        
    print("mixture (non-zero dof)="+ str(mix.mixture) + "\n")
    print("dof="+str(res["dof"]) + "\n")
    print("scale="+str(res["scale"]) + "\n")
    import ipdb; ipdb.set_trace()

def lrt_method(stat,dof):
    '''
    Standard way to compute p-value when no boundary conditions
    '''
    pv = (st.chi2.sf(stat,dof)) 
    return pv

def linreg_uniscan(X,y,covar=None,REML=False):
    '''
    Iterate through each column of X, using it in the alternative model, and otherwise, 
    only covar/covar+bias in the null model. Y is a 1D phenotype vector
    Output:
        p-value, LLnull, LLalt
    '''
    numtests=X.shape[1]
    print("numtests=" + str(numtests))
    pv = np.nan*np.ones(numtests)
    m_alt=[]
    nullfeat = np.ones_like(y) #bias term
    if covar is not None:
        nullfeat=np.hstack((nullfeat,covar))
        
    m_null = linreg(nullfeat,y,REML=REML)
    ttt0=time.time()
    for i in range(numtests):
        snp = X[:,i:i+1]
        if np.isnan(y).sum()>0: raise Exception("missing data found")
        snp_w_nullfeat = np.hstack((snp,nullfeat))
        m_alt.append(linreg(snp_w_nullfeat,y,REML=REML))
        pv[i] = lrt(-2*(m_alt[i]['nLL']-m_null['nLL']),dof=1)
        if i % 50000==0:
            ttt1=time.time()
            print("Elapsed time for %d of %d tests is %.2f seconds" % (i, numtests,(ttt1-ttt0)))       
    return pv,m_null,m_alt

def linreg(X,y,REML=True,**kwargs):
        '''
        Perform linear regression using the built-in least squares solver
        --------------------------------------------------------------------------
        Output dictionary:
        'nLL'       : negative log-likelihood
        'beta'      : [D*1] array of fixed effects weights beta
        'sigma2'        : variance of noise
        'REML'      : True: REML was computed, False: ML was computed
        --------------------------------------------------------------------------
        '''

        lsqSol=la.lstsq(X,y)

        beta=lsqSol[0] #weights
        r2=lsqSol[1] #squared residuals
        D=lsqSol[2]  #rank of design matrix

        N=y.shape[0]

        if not(REML):
            sigma2 = r2/(N)
            nLL =  N*0.5*sp.log(2*sp.pi*sigma2) + N*0.5
        else:
            sigma2 = r2 / (N-D)
            nLL = N*0.5*sp.log(2*sp.pi*sigma2) + 0.5/sigma2*r2;
            nLL -= 0.5*D*sp.log(2*sp.pi*sigma2);#REML term

        result = {
                'nLL':nLL,
                'sigma2':sigma2,
                'beta':beta,
                'REML':REML
                }        
        if result['nLL'] is None: raise Exception("no nLL result")
        return result

def stats(a):
    '''
    Reports the mean, std, min, and max values in a
    '''
    print("mean="+str(np.mean(a)) + ", std=" +str(np.std(a)) + ", [min,max]=[" + str(np.min(a)) + "," + str(np.max(a)) + "]")



