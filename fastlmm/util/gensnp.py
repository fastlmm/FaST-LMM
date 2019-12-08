from __future__ import absolute_import
import scipy as sp
import pdb
import scipy.linalg as la

def gendat(TWO_KERNEL,nInd,nSnp,nCovar,minMaf=0.05,maxMaf=0.4,minSigE2=0.5,maxSigE2=1,minSigG2=0.5,maxSigG2=1):
    '''
    Generate synthetic SNPs and phenotype.
    SNPs are iid, and there is no population structure.
    Phenotype y is generated from a LMM with SNPs in a PS kernel.

    Returns:
        covDat
        y
        psSnps
    '''

    if TWO_KERNEL:
        psSnps=gensnps(nInd,nSnp,minMaf,maxMaf)
        psK=psSnps.dot(psSnps.T)
        psK+=1e-5*sp.eye(nInd)
        psKchol=la.cholesky(psK)
    else:
        psSnps=None

    covDat=sp.random.uniform(0,1,(nInd,nCovar))
    covWeights=sp.random.uniform(-0.5,0.5,(nCovar,1))

    sigE2=sp.random.uniform(low=minSigE2,high=maxSigE2)
    sigG2=sp.random.uniform(low=minSigG2,high=maxSigG2)

    ##generate the phenotype using the background kernel and covariates
    if TWO_KERNEL:
        y_pop=sp.sqrt(sigG2)*psKchol.dot(sp.randn(nInd,1))
    else:
        y_pop=0
    y_noise=sp.randn(nInd,1)*sp.sqrt(sigE2)
    y=(covDat.dot(covWeights) + y_pop + y_noise).squeeze()
    return covDat, y, psSnps

def gensnps(numSnp,nInd,minMaf=0.05,maxMaf=0.4,standardize=False):
    '''
    Generate independent SNPs, from independent people (no population structure)
    '''

    if (minMaf<0)|(minMaf>1): raise Exception('invalid minMaf provided')
    if (maxMaf<0)|(maxMaf>1): raise Exception('invalid minMaf provided')
    if (maxMaf<minMaf): raise Exception('invalid minMaf/maxMaf combo provided')

    maf=sp.random.uniform(low=minMaf,high=maxMaf,size=(numSnp,1))

    numChrm=2
    snpDat=sp.zeros((numSnp,nInd))
    for ch in sp.arange(0,numChrm):
        mafRep=sp.tile(maf,(1,nInd))
        snpDat+=sp.random.uniform(size=(numSnp,nInd))<mafRep

    if standardize:
        raise NotImplementedError()

    return snpDat
