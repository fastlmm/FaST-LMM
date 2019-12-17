"""
example of how to use feature selection from python (see also command line interface)
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import scipy.linalg as LA
import sys, os
from fastlmm.pyplink.snpreader.Bed import Bed
from fastlmm.feature_selection import PerformSelectionDistributable as dist

if __name__ == '__main__':
    #filepath = r"\\erg00\Genetics\synthetic\alkes2013\large"
    filepath = r"\\erg00\genetics\dbgap\aric"
    files = [r"autosomes"]

    numpcs = [int(arg) for arg in sys.argv[1:]]

    #files = [r"psz.0",r"fst.0025p.005\psz.0",r"fst.0025p.05\psz.0",r"fst.005p.05\psz.0"]
    #files = [r"fst.0025p.005\psz.0",r"fst.0025p.05\psz.0",r"fst.005p.05\psz.0"]
    

    for file in files:
        computePC(file, filepath = filepath, numpc = numpcs)
def getEigvecs_fn(fn,numpcs):
    fnout = "%s_pc%i.vecs"%(fn,numpcs)
    return fnout
def computePC(file, filepath = None, numpc = [5]):
    if filepath is not None:
        fn = os.path.join(filepath,file)
    else:
        fn = file
    if type(numpc) is int or type(numpc) is float:
        numpc = [numpc]
    alt_snpreader = Bed(fn)
    print("computing K")
    K = dist.build_kernel_blocked(fn,alt_snpreader=alt_snpreader)
    print("computing the Eigenvalue decomposition of K")
    [s_all,u_all] = LA.eigh(K)
    s_all=s_all[::-1]
    u_all=u_all[:,::-1]
    for numpcs in numpc:
        #import pdb; pdb.set_trace()
        print("saving %i PCs from %s" %(numpcs,fn))
        
        #import pdb; pdb.set_trace()

        s=s_all[0:numpcs]
        u = u_all[:,0:numpcs]
        outu = np.zeros((u_all.shape[0],numpcs+2),dtype = "|S20")
        outu[:,0:2] = alt_snpreader.original_iids
        outu[:,2::]=u
        fnout = getEigvecs_fn(fn,numpcs)
    
        np.savetxt(fnout,outu,fmt="%s",delimiter = "\t")
        fnout = "%s_pc%i.vals"%(fn,numpcs)
        #outs = np.zeros((s.shape[0],u.shape[1]+2),dtype = "|S20")
        np.savetxt(fnout,s,fmt="%.5f",delimiter = "\t")
    return s_all,u_all