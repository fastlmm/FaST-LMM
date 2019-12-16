from __future__ import absolute_import
from fastlmm.feature_selection import PerformSelectionDistributable as psd
import fastlmm.util.preprocess as up
import numpy as np

def set_Gexclude(G_exclude, G1, i_exclude):
        if G_exclude is None and i_exclude is not None and i_exclude.sum()>0:
            assert self.G0 is not None, "i_exclude without SNPs to exclude"
            G = np.hstack((self.G0[:,i_exclude],G1))
            i_G1=np.ones(G.shape[1],dtype='bool')
            n_exclude = i_exclude.sum()
            i_G1[0:n_exclude]=False
        elif G_exclude is not None:
            G = np.hstack((G_exclude,G1))
            i_G1=np.ones(G.shape[1],dtype='bool')
            n_exclude = G_exclude.shape[1]
            i_G1[0:n_exclude]=False
        else:
            G = G1
            i_G1=np.ones(G.shape[1],dtype='bool')
            n_exclude = 0
        return G, i_G1, n_exclude

def set_snps0(SNPs0,sample_size,i_exclude=None, forcefullrank=False,blocksize=10000):
    '''
    In full rank case, loads up the SNPs in blocks, and construct the kernel.
    In low rank case, loads up all SNPs in to memory
    '''    
    if SNPs0 is None:
        return None, None
    if "K" in SNPs0:
        K0 = SNPs0["K"]
        G0 = None
    elif "data" in SNPs0:
        K0 = None
        G0 = SNPs0["data"]["snps"]
    else:        
        #full rank
        if len(SNPs0["snp_set"]) > sample_size or forcefullrank:# N = Y.shape[0]                      
            SNPs0["K"] = psd.build_kernel_blocked(snpreader=SNPs0["reader"], snp_idx=SNPs0["snp_set"].to_index, blocksize=blocksize,allowlowrank=forcefullrank)
            K0 = SNPs0["K"]
            G0 = None
        else:
            #low rank            
            K0 = None
            SNPs0["data"] = SNPs0["snp_set"].read()
            SNPs0["data"]["snps"] = up.standardize(SNPs0["data"]["snps"])
            G0 = SNPs0["data"]["snps"]

    #lrt_up should never do exclusion, because set_snps0 should only get called once, in run_once, without exclusion
    #exclude. So this is only for score test and lrt. 
    if i_exclude is not None:
        if K0 is not None:
            #Also note in the full rank case with exclusion, for score, one could in principle use low rank updates to make it faster,
            #when the number of excluded SNPs is small: it wold be cubic in num_excluded * num_inner*num_outer iterations, versus now
            #where it is cubic in N in the outer loop only once
            K_up = psd.build_kernel_blocked(snpreader=SNPs0["reader"], snp_idx=np.array(SNPs0["snp_set"].to_index)[i_exclude], blocksize=blocksize,allowlowrank=forcefullrank)
            K0 = K0 - K_up
        elif G0 is not None:
            G0 = G0[:,~i_exclude]                        
        num_snps = SNPs0["num_snps"] - i_exclude.sum()
    else:
        num_snps = SNPs0["num_snps"]
    #intersect data?
        
    #normalize:
    if K0 is not None:
        K0 = K0 / num_snps#K0.diagonal().mean()
    elif G0 is not None:
        G0 = G0 / np.sqrt( num_snps )#(G0*G0).mean() ) # computes the sqrt of the mean of the diagonal of K=GG^T; *  means pointwise multiplication 
    return G0, K0