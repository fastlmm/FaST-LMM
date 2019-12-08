from __future__ import absolute_import
import copy
import pdb
import scipy.linalg as LA
import scipy as SP
import numpy as NP
import logging as LG
import scipy.optimize as opt
import scipy.stats as ST
import scipy.special as SS
import os
import sys
from fastlmm.pyplink.plink import *
from pysnptools.util.pheno import *
from fastlmm.util.mingrid import *
from fastlmm.util.util import *
import fastlmm.util.stats as ss
import fastlmm.inference as inference
import fastlmm.association.score as score
import fastlmm.association as association
import statsmodels.api as sm

from sklearn import linear_model
from six.moves import range

class lrt(association.varcomp_test):
    __slots__ = ["model0","model1","lrt","forcefullrank","nullModel","altModel","G0","K0","__testGcalled"]

    def __init__(self,Y,X=None,model0=None,appendbias=False,forcefullrank=False,
                 G0=None,K0=None,nullModel=None,altModel=None):
        association.varcomp_test.__init__(self,Y=Y,X=X,appendbias=appendbias)
        N = self.Y.shape[0]
        self.forcefullrank=forcefullrank

        self.nullModel = nullModel
        self.altModel = altModel
        self.G0=G0
        self.K0=K0
        self.__testGcalled=False
        if ('penalty' not in nullModel) or nullModel['penalty'] is None:
            nullModel['penalty'] = 'l2'
        if nullModel['effect']=='fixed':

            if nullModel['link']=='linear':
                self._nullModelLinReg(G0)
            elif nullModel['link']=='logistic':
                self._nullModelLogReg(G0, nullModel['penalty'])
            else:
                assert False, 'Unknown link function.'
            assert 'approx' not in nullModel or nullModel['approx'] is None, 'Cannot use approx with fixed effect'


        elif nullModel['effect']=='mixed':
            if nullModel['link']=='linear':
                self._nullModelMixedEffectLinear(G0=G0,K0=K0)
            else:
                self._nullModelMixedEffectNonLinear(G0, nullModel['approx'], nullModel['link'], nullModel['penalty'])
        else:
            assert False, 'Unknown effect type.'

    def _nullModelLogReg(self, G0, penalty='L2'):        
        assert G0 is None, 'Logistic regression cannot handle two kernels.'
        self.model0={}       
        import statsmodels.api as sm 
        logreg_mod = sm.Logit(self.Y,self.X)
        #logreg_sk = linear_model.LogisticRegression(penalty=penalty)

        logreg_result = logreg_mod.fit(disp=0)        
        self.model0['nLL']=logreg_result.llf
        self.model0['h2']=SP.nan   #so that code for both one-kernel and two-kernel prints out
        self.model0['a2']=SP.nan

    def _nullModelLinReg(self, G0):
        assert G0 is None, 'Linear regression cannot handle two kernels.'
        self.model0={}
        model = ss.linreg(self.X,self.Y)        
        self.model0['h2']=SP.nan   #so that code for both one-kernel and two-kernel prints out
        self.model0['nLL']=model['nLL']           
    
    def _nullModelMixedEffectLinear(self, G0=None,K0=None):
        lmm0 = inference.getLMM(forcefullrank = self.forcefullrank)
        if G0 is not None:
            lmm0.setG(G0=G0,K0=K0)
        lmm0.setX(self.X)
        lmm0.sety(self.Y)
        self.model0 = lmm0.findH2()# The null model only has a single kernel and only needs to find h2

    def _nullModelMixedEffectNonLinear(self, G0, approx, link, penalty):
        if G0 is None:
            return self._nullModelMixedEffectNonLinear1Kernel(approx, link, penalty)
        return self._nullModelMixedEffectNonLinear2Kernel(G0, approx, link, penalty)

    def _nullModelMixedEffectNonLinear1Kernel(self, approx, link, penalty):
        if self.forcefullrank:
            assert False, "Not implemented yet."
        else:
            glmm0 = inference.getGLMM(approx, link, self.Y, None, None, penalty=penalty)
        glmm0.setX(self.X)
        glmm0.sety(self.Y)
        glmm0.optimize()
        self.model0 = {}        
        self.model0['h2']=0.0
        self.model0['a2']=NP.nan
        self.model0['nLL']=-glmm0.marginal_loglikelihood()
        self.model0['sig02'] = glmm0.sig02
        self.model0['sig12'] = glmm0.sig12
        self.model0['sign2'] = glmm0.sign2
        for i in range(len(glmm0.beta)):
            self.model0['beta' + str(i)] = glmm0.beta[i]

    def _nullModelMixedEffectNonLinear2Kernel(self, G0, approx, link, penalty):
        if self.forcefullrank:
            assert False, "Not implemented yet."
        else:
            glmm0 = inference.getGLMM(approx, link, self.Y, G0, None, penalty=penalty)
        glmm0.setX(self.X)
        glmm0.setG(G0)
        glmm0.sety(self.Y)
        glmm0.optimize()
        self.model0 = {}
        
        if glmm0.sig02 + glmm0.sign2 <= NP.sqrt(NP.finfo(NP.float).eps):
            h2 = NP.nan
        else:
            h2 = glmm0.sig02 / (glmm0.sig02 + glmm0.sign2)

        self.model0['h2']=h2
        self.model0['a2']=0.0
        self.model0['nLL']=-glmm0.marginal_loglikelihood()
        self.model0['sig02'] = glmm0.sig02
        self.model0['sig12'] = glmm0.sig12
        self.model0['sign2'] = glmm0.sign2
        for i in range(len(glmm0.beta)):
            self.model0['beta' + str(i)] = glmm0.beta[i]

    def testGupdate(self, y, X, type=None):         
        '''
        Assume that testG has already been called (and therefore the
        expensive part of SVD related to the test SNPs), and that we are only changing 
        the phenotype and covariates (e.g. for permutations).
        Recomputes the null model, and, crucially, cheaply, the alternative model
        '''       
        assert self._testGcalled, "must have called testG before updateTestG which assumes only a change in y"                
        origX=self.X
        origY=self.Y
        self._updateYX(y,X)
  
        #don't need this, as is invariant under permutations of only test SNPs or compliment          
        #if self.nullModel['effect']=='fixed' and self.nullModel['link']=='linear':
        #    self._nullModelLinReg(self.G0)
        #else:
        #    raise Exception("not implemented")

        #compute the alternative likelihood
        if self.altModel['effect']=='fixed':
            raise Exception("not implemented")            
        elif self.altModel['effect']=='mixed' and self.altModel["link"]=="linear":
            (lik1,stat,alteqnull) = self._altModelMixedEffectLinearUpdate(self.model1)                   
        else:
            raise Exception("not implemented")
                   
        #due to optimization the alternative log-likelihood might be a about 1E-6 worse than the null log-likelihood 
        pvreg = (ST.chi2.sf(stat,1.0)) #standard way to compute p-value when no boundary conditions
        if SP.isnan(pvreg) or pvreg>1.0:
            pvreg=1.0                
        pv = 0.5*pvreg                  #conservative 50/50 estimate
        if alteqnull: pv=1.0            #chi_0 component
                    
        test={
              'pv':pv,
              'stat':stat,
              'lik1':lik1,
              'lik0':self.model0,
              'alteqnull':alteqnull
              }
        self._updateYX(origY,origX)
        return test

    @property
    def _testGcalled(self):
        return self.__testGcalled

    def testG(self, G1, type=None,i_exclude=None,G_exclude=None):
        """
        Params:
            G1:         SNPs to be tested
            type:       Dummy
            i_exclude:  Dummy
            G_exclude:  Dummy
        """    
        self.__testGcalled=True
        #compute the alternative likelihood
        if self.altModel['effect']=='fixed':
            if self.altModel['link']=='linear':
                (lik1,stat,alteqnull) = self._altModelLinReg(G1)
            elif self.altModel['link']=='logistic':
                assert False, 'Link function not implemented yet.'
            else:
                assert False, 'Unkown link function.'
            assert 'approx' not in altModel or altModel['approx'] is None, 'Cannot use approx with fixed effect'
        elif self.altModel['effect']=='mixed':
            if self.altModel['link']=='linear':
                (lik1,stat,alteqnull) = self._altModelMixedEffectLinear(G1)
            else:
                (lik1,stat,alteqnull) = self._altModelMixedEffectNonLinear(G1, self.altModel['approx'],
                                                                           self.altModel['link'],
                                                                           self.altModel['penalty'])
        else:
            assert False, 'Unkown effect type.'        
                   
        #due to optimization the alternative log-likelihood might be a about 1E-6 worse than the null log-likelihood 
        pvreg = (ST.chi2.sf(stat,1.0)) #standard way to compute p-value when no boundary conditions
        if SP.isnan(pvreg) or pvreg>1.0:
            pvreg=1.0                
        pv = 0.5*pvreg                  #conservative 50/50 estimate
        if alteqnull: pv=1.0            #chi_0 component
                    
        test={
              'pv':pv,
              'stat':stat,
              'lik1':lik1,
              'lik0':self.model0,
              'alteqnull':alteqnull
              }
        return test

    def _altModelLinReg(self, G1):
        assert False, 'Not implemented yet.'
    
    def _altModelMixedEffectLinearUpdate(self, lmm1, tol=0.0):
        '''
        Assumes that setG has been called already (expensive in many cases), and does not redo it.
        '''        
        if self.G0 is not None:
            raise Exception("not implemented")
        else:               
            lmm1.setX(self.X)
            lmm1.sety(self.Y)
            lik1 = lmm1.findH2()#The alternative model has one kernel and needs to find only h2            
            alteqnull=lik1['h2']<=(0.0+tol)
        stat = 2.0*(self.model0['nLL'] - lik1['nLL'])        
        self.model1=lmm1
        return (lik1,stat,alteqnull)

    def _altModelMixedEffectLinear(self, G1,tol=0.0):
        lmm1 = inference.getLMM(forcefullrank = self.forcefullrank)        
        if self.G0 is not None:
            lmm1.setG(self.G0, G1)
            lmm1.setX(self.X)
            lmm1.sety(self.Y)
            lik1 = lmm1.findA2()#The alternative model has two kernels and needs to find both a2 and h2
            alteqnull=lik1['a2']<=(0.0+tol)
        else:
            lmm1.setG(G1)
            lmm1.setX(self.X)
            lmm1.sety(self.Y)
            lik1 = lmm1.findH2()#The alternative model has one kernel and needs to find only h2            
            alteqnull=lik1['h2']<=(0.0+tol)
        stat = 2.0*(self.model0['nLL'] - lik1['nLL'])        
        self.model1=lmm1
        return (lik1,stat,alteqnull)

    def _altModelMixedEffectNonLinear(self, G1, approx, link, penalty):
        if self.G0 is None:
            (lik1,stat) = self._altModelMixedEffectNonLinear1Kernel(G1, approx, link, penalty)
        else:
            (lik1,stat) = self._altModelMixedEffectNonLinear2Kernel(G1, approx, link, penalty)

        if stat < 1e-4:
            lik1['nLL'] = self.model0['nLL']
            lik1['h2'] = self.model0['h2']
            lik1['a2'] = self.model0['a2']
            stat = 0.0
            alteqnull = True
        else:
            alteqnull = False

        return (lik1,stat,alteqnull)

    def _altModelMixedEffectNonLinear1Kernel(self, G1, approx, link, penalty):
        if self.forcefullrank:
            assert False, 'Not working, call Danilo'
            assert False, "Not implemented yet."
        else:
            glmm1 = inference.getGLMM(approx, link, self.Y, G1, None, penalty=penalty)

        glmm1.setX(self.X)

        glmm1.sety(self.Y)

        glmm1.setG(G1)
        glmm1.optimize()

        assert glmm1.sig02 >= 0.0 and glmm1.sign2 >= 0

        if glmm1.sig02 + glmm1.sign2 <= NP.sqrt(NP.finfo(NP.float).eps):
            h2 = NP.nan
        else:
            h2 = glmm1.sig02 / (glmm1.sig02 + glmm1.sign2)

        a2 = NP.nan

        lik1 = {'nLL':-glmm1.marginal_loglikelihood(),
                'h2':h2,
                'a2':a2}

        lik1['sig02'] = glmm1.sig02
        lik1['sig12'] = glmm1.sig12
        lik1['sign2'] = glmm1.sign2
        for i in range(len(glmm1.beta)):
            lik1['beta' + str(i)] = glmm1.beta[i]

        stat = 2.0*(self.model0['nLL'] - lik1['nLL'])
        return (lik1,stat)

    def _altModelMixedEffectNonLinear2Kernel(self, G1, approx, link, penalty):
        if self.forcefullrank:
            assert False, "Not implemented yet."
        else:
            glmm1 = inference.getGLMM(approx, link, self.Y, self.G0,
                                                 G1, penalty=penalty)

        glmm1.setX(self.X)

        glmm1.sety(self.Y)

        glmm1.setG(self.G0, G1)
        glmm1.optimize()

        assert glmm1.sig02 >= 0.0 and glmm1.sig12 >= 0.0 and glmm1.sign2 >= 0


        if glmm1.sig02 + glmm1.sig12 + glmm1.sign2 <= NP.sqrt(NP.finfo(NP.float).eps):
            # in this case we don't have enough precision to calculate the
            # proportion between sig02+sig12 and the total or it does not make sense
            # because the covariance of the posterior tends to zero
            h2 = NP.nan
        else:
            h2 = (glmm1.sig02+glmm1.sig12) / (glmm1.sig02 + glmm1.sig12 + glmm1.sign2)

        if glmm1.sig02 + glmm1.sig12 <= NP.sqrt(NP.finfo(NP.float).eps):
            a2 = NP.nan
        else:
            a2 = glmm1.sig12 / (glmm1.sig02+glmm1.sig12)

        lik1 = {'nLL':-glmm1.marginal_loglikelihood(),
                'h2':h2,
                'a2':a2}

        lik1['sig02'] = glmm1.sig02
        lik1['sig12'] = glmm1.sig12
        lik1['sign2'] = glmm1.sign2
        for i in range(len(glmm1.beta)):
            lik1['beta' + str(i)] = glmm1.beta[i]

        stat = 2.0*(self.model0['nLL'] - lik1['nLL'])
        return (lik1,stat)
