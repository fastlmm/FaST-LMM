from __future__ import absolute_import
import scipy as sp
import pdb
import scipy.linalg as la
import fastlmm.util.util as utilx
import sys

def genphen(y_G0,G1,covDat,options,nInd,K1=None,fracCausal=None,randseed=None):
    '''
    Generate synthetic phenotype with a LMM and linear kernels, using SNPs in G1 for signal,
    snps in GO for background, and one of two link functions.
    If genlink=='linear', uses linear LMM. If genlink='logistic', then thresholds to get binary.

    fracCausal is the fraction of SNPs that are causal (rounding up) when G1 is provided
    
    Only one of G1 and K1 can be not None (G1 is good for low rank, K1 for full rank)
    
    Returns:    
        y  (binary, or real-valued, as dictated by genlink)      
        If y is binary, casefrac are 1s, and the rest 0s (default casefrac=0.5)
    Notes: uses sp.random.X so that the seed that was set can be used
    '''    
    
    sp.random.seed(int(randseed % 2147483647)) #old maxint

    if "numBackSnps" in options and options["numBackSnps"]>0:
        raise Exception("I accidentally deleted this move from FastLMmSet to here, see code for FastLmmSet.py from 11/24/2013")
    
    ## generate from the causal (not background) SNPs---------------
    assert not (G1 is not None and K1 is not None), "need to provide only either G1 or K1"
    fracCausal=options['fracCausal']    

    if G1 is not None and options["varG"]>0:    
        if fracCausal>1.0 or fracCausal<0.01: raise Exception("fraCausal should be between 0.01 and 1")    
        nSnp=G1.shape[1]        
        if fracCausal !=1.0:
            nSnpNew=sp.ceil(fracCausal*nSnp)            
            permutationIndex = utilx.generate_permutation(sp.arange(0,nSnp),randseed)[0:nSnpNew]            
            G1new=G1[:,permutationIndex]
        else:     
            nSnpNew=nSnp       
            G1new=G1
    elif K1 is not None:
        assert(fracCausal==1.0 or fracCausal is None)
        pass
    else:
        assert options['varG']==0, "varG is not zero, but neither G1 nor K1 were provided"
    
    stdG=sp.sqrt(options['varG'])

    if stdG>0:
        if G1 is not None:
            y_G1=stdG*G1new.dot(sp.random.randn(nSnpNew,1))    #good for low rank
        else:
            K1chol = la.cholesky(K1)
            y_G1=stdG*K1chol.dot(sp.random.randn(nInd,1))       #good for full rank
    else:
        y_G1=0.0
   ##----------------------------------------------------------------

    if covDat is not None: 
        nCov=covDat.shape[1]    
        covWeights=sp.random.randn(nCov, 1)*sp.sqrt(options['varCov'])
        y_beta=covDat.dot(covWeights)
    else:
        y_beta=0.0
           
    y_noise_t=0    
    #heavy-tailed noise 
    if options['varET']>0:        
        y_noise_t=sp.random.standard_t(df=options['varETd'],size=(nInd,1))*sp.sqrt(options['varET'])          
    else:
        y_noise_t=0
    
    #gaussian noise
    y_noise=sp.random.randn(nInd,1)*sp.sqrt(options['varE'])  
                       
    y=y_noise + y_noise_t + y_G0 + y_beta + y_G1   
    y=y[:,0]#y.flatten()          

   
    if options['link']=='linear':
        return y
    elif options['link']=='logistic':
        if options['casefrac'] is None: options['casefrac']=0.5        
        ysort=sp.sort(y,axis=None)
        thresh=ysort[sp.floor(nInd*options['casefrac'])]
        ybin=sp.array(y>thresh,dtype="float")
        return ybin
    else:
        raise Exception("Invald link function for data generation")
