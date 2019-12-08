from __future__ import absolute_import
import fastlmm.association as association
import fastlmm.association.lrt as lrt
import numpy as np
import scipy.linalg as la

class lrt_fixed(lrt.lrt): #!!!what is this for?

    def __init__(self, Y, X=None, model0 = None, appendbias=False, forcefullrank = False, G0 = None, nullModel = None):
        lrt.lrt.__init__(self,Y=Y, X=X, model0 = model0, appendbias=appendbias, forcefullrank = forcefullrank, G0 = G0, nullModel = nullModel)

    def testG(self, G1, type=None, altModel=None, dof = 100000000, meanG=False):
        #compute the alternative likelihood
        if dof<G1.shape[1]:
            [u,s,v] = la.svd(G1) #!!!use big_svd?
            G1 = u[:,0:dof]
        elif meanG:
            G1 = G1.mean(1)[:,np.newaxis]
        if altModel['name']=='linreg':
            (lik1,stat) = self._altModelLinReg_fixed(G1)
        elif altModel['name']=='logitreg':
            assert False, 'Null model not implemented yet.'
        elif altModel['name']=='probitreg':
            assert False, 'Null model not implemented yet.'
        elif altModel['name']=='lmm':
            assert False, 'Null model not implemented yet.'
            (lik1,stat) = self._altModelLMM_fixed(G1)
        elif altModel['name']=='glmm':
            assert False, 'Null model not implemented yet.'
            (lik1,stat) = self._altModelGLMM_fixed(G1, altModel['approx'], altModel['link'])
        else:
            assert False, 'Unrecognized alt model.'

        #analytical P-value assuming a 50-50 mixture of Chi2_0 and Chi2_1 distributions
        pv = (ST.chi2.sf(stat,G1.shape[1]))
        if SP.isnan(pv) or pv>0.5:
            pv=1.0 #due to optimization the alternative log-likelihood might be a about 1E-6 worse than the null log-likelihood
        test={
              'pv':pv,
              'stat':stat,
              'type':type, #!! is it OK to have an object here instead of a name?
              'lik1':lik1
              }
        return test

    def _altModelLinReg_fixed(self, G1):
        assert self.model0['G0'] is None, 'Linear regression cannot handle two kernels.'
        X = np.concatenate((self.X,G1),1)
        model1=ss.linreg(X,self.Y,REML=False)
        lik1 = self.model1['model']['nLL']
        stat = 2.0*(lik1 - self.model0['nLL'])
        return lik1,stat

    def _altModelLMM_fixed(self, G1):
        X = np.concatenate((self.X,G1),1)
        self.model0['model'].setX(X)
        model1 = self.model0['model'].findH2()# The null model only has a single kernel and only needs to find h2
        lik1 = self.model0['model']['nLL']
        stat = 2.0*(lik1 - self.model0['nLL'])
        return lik1,stat