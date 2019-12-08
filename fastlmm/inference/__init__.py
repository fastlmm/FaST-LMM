from __future__ import absolute_import
from fastlmm.inference.fastlmm_predictor import FastLMM
from fastlmm.inference.linear_regression import LinearRegression


#from bin2kernel import Bin2Kernel
#from bin2kernel import makeBin2KernelAsEstimator
#from bin2kernel import Bin2KernelLaplaceLinearN
#from bin2kernel import getFastestBin2Kernel
#from bin2kernel import Bin2KernelEPLinearN

from .laplace import LaplaceGLMM_N3K1, LaplaceGLMM_N1K3
from .ep import EPGLMM_N3K1, EPGLMM_N1K3
from .lmm import LMM

'''
Return the fastest implementation according to the data provided.
It basically determines if the number of individuals is bigger than
the number of snps.
'''
def getGLMM(approx, link, y, G0, G1, penalty=None, penalizeBias=False, debug=False):
    k = 0
    if G0 is not None:
        k += G0.shape[1]
    if G1 is not None:
        k += G1.shape[1]

    N = y.size
    if N <= k:
        if approx == 'laplace':
            return LaplaceGLMM_N3K1(link, penalty=penalty, penalizeBias=penalizeBias, debug=debug)
        if approx == 'ep':
            return EPGLMM_N3K1(link, penalty=penalty, penalizeBias=penalizeBias, debug=debug)
        assert False, 'Unkown approximation.'

    if approx == 'laplace':
        return LaplaceGLMM_N1K3(link, penalty=penalty, penalizeBias=penalizeBias, debug=debug)
    if approx == 'ep':
        return EPGLMM_N1K3(link, penalty=penalty, penalizeBias=penalizeBias, debug=debug)

    assert False, 'Unkown approximation.'

def getLMM(forcefullrank=False):
    return LMM(forcefullrank=forcefullrank)
