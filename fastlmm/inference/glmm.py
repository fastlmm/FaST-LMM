from __future__ import absolute_import
import scipy as SP
import scipy.stats as ST
import numpy as NP
from numpy import dot
from scipy.linalg import cholesky,solve_triangular
from fastlmm.external.util.math import check_definite_positiveness,check_symmetry,ddot,dotd,trace2
from fastlmm.external.util.math import stl, stu
from sklearn.base import BaseEstimator
from fastlmm import Pr
import sys
from six.moves import range

'''
    Important! Always run test.py in the current folder for unit testing after
    changes have been made.
'''

class GLMM(object):
    '''
    generalized linear mixed model having up to two linear kernels
    f ~ N(X*beta, sig02*K0 + sig12*K1 + sign2*I),
    y ~ Bern(link(f))
    where
    K0 = G0*G0^T
    K1 = G1*G1^T
    '''

    def __init__(self, penalty=None, penalizeBias=False, debug=False):
        '''
        Input:
        approx        : 'laplace' or 'ep'
        link          : 'logistic' or 'erf'
        penalty       : None, 'l1' or 'l2'
        penalizeBias  : True or False
        '''
        self._debug = debug
        self._X=None
        self._y=None
        self._sig02=0.0
        self._sig12=0.0
        self._sign2=0.0
        self._G0 = None
        self._G1 = None
        self._isK0Set=False
        self._isK1Set=False
        self._beta=None
        self._mapIn2OutY=None
        self._mapOut2InY=None
        self._outYType=None
        assert penalty in set([None,'l1','l2'])
        assert penalty != 'l1', 'l1 penalizer is not fully implemented yet.'
        self._penalty = penalty
        self._penalizeBias = penalizeBias
        self._hasBias = False
        self._biasIndex = None
        # penalization weights
        self._lambdaBeta = 0.1
        self._lambdaSig02 = 0.1
        self._lambdaSig12 = 0.1
        self._lambdaSign2 = 0.1

        self._updateConstantsCount = 1
        self._updateApproximationCount = 1

    def setX(self, X):
        '''
        set the fixed effects X (covariates).
        --------------------------------------------------------------------------
        Input:
        X       : [N*D] 2-dimensional array of covariates
        --------------------------------------------------------------------------
        '''
        assert type(X) is NP.ndarray, 'X must be a numpy.ndarray.'
        assert NP.all(NP.isfinite(X) & ~NP.isnan(X)), 'X must have only numbers.'
        assert len(X.shape)==2, 'X must be a 2-dimensional array.'
        assert X.shape[0] > 0
        assert X.shape[1] > 0
        self._X = X
        self._updateConstantsCount += 1

        self._hasBias = False
        for i in range(X.shape[1]):
            if len(NP.unique(X[:,i])) == 0 and X[0,i] == 1.0:
                self._hasBias = True
                self._biasIndex = i
                break

    def sety(self, y):
        '''
        set the phenotype y.
        --------------------------------------------------------------------------
        Input:
        y       : [N] 1-dimensional array of phenotype values (-1.0 or 1.0)
        --------------------------------------------------------------------------
        '''
        assert type(y) is NP.ndarray, 'y must be a numpy.ndarray.'
        assert len(y.shape)==1, 'y must be a 1-dimensional array.'
        assert y.shape[0] > 0

        uniquey = list(set(y))
        assert len(uniquey)==2, 'y must have two unique values.'
        uniquey = sorted(uniquey)

        self._mapIn2OutY = {-1.0:uniquey[0],+1.0:uniquey[1]}
        self._mapOut2InY = {uniquey[0]:-1.0,uniquey[1]:+1.0}
        self._outYType = y.dtype

        self._y = NP.empty(y.shape[0])
        self._y[y==uniquey[0]] = -1.0
        self._y[y==uniquey[1]] = +1.0

        self._updateConstantsCount += 1

    def setG(self, G0, G1=None):
        '''
        set the kernels K0 and K1 from G0 and G1.
        ----------------------------------------------------------------------------
        Input:
        G0              : [N*k0] array of random effects
        G1              : [N*k1] array of random effects (optional)
        -----------------------------------------------------------------------------
        '''
        assert type(G0) is NP.ndarray, 'G0 must be a numpy.ndarray.'
        assert NP.all(NP.isfinite(G0) & ~NP.isnan(G0)), 'G0 must have only numbers.'
        assert len(G0.shape)==2, 'G0 must be a 2-dimensional array.'
        assert G0.shape[0] > 0
        assert G0.shape[1] > 0
        self._G0 = G0

        if G1 is not None:
            assert type(G1) is NP.ndarray, 'G1 must be a numpy.ndarray.'
            assert NP.all(NP.isfinite(G1) & ~NP.isnan(G1)), 'G1 must have only numbers.'
            assert len(G1.shape)==2, 'G1 must be a 2-dimensional array.'
            assert G1.shape[0] > 0
            assert G1.shape[1] > 0
            self._G1 = G1

        self._isK0Set = G0 is not None
        self._isK1Set = G1 is not None

        self._updateConstantsCount += 1

    # Operation dot(A,K).
    def _ldotK(self, A):
        pass

    # Operation dot(K,A).
    def _rdotK(self, A):
        pass

    # Return K's diagonal.
    def _dKn(self):
        pass

    def printDebug(self):
        pass

    '''
    -------------------------------------------------------------------------------
    Begin of hyperparameters
    '''
    @property
    def sig02(self):
        return self._sig02

    @sig02.setter
    def sig02(self, v):
        assert NP.isscalar(v)
        assert NP.isfinite(v)
        if self._sig02 == v:
            return
        self._updateApproximationCount += 1
        self._sig02 = v

    @property
    def sig12(self):
        return self._sig12

    @sig12.setter
    def sig12(self, v):
        assert NP.isscalar(v)
        assert NP.isfinite(v)
        if self._sig12 == v:
            return
        self._updateApproximationCount += 1
        self._sig12 = v

    @property
    def sign2(self):
        return self._sign2

    @sign2.setter
    def sign2(self, v):
        assert NP.isscalar(v)
        assert NP.isfinite(v)
        if self._sign2 == v:
            return
        self._updateApproximationCount += 1
        self._sign2 = v

    @property
    def beta(self):
        return self._beta.copy()

    @beta.setter
    def beta(self, v):
        assert NP.all(NP.isfinite(v))
        if self._beta is not None and NP.all(self._beta == v):
            return
        self._updateApproximationCount += 1
        self._beta = v.copy()
    '''
    End of hyperparameters
    -------------------------------------------------------------------------------
    '''

    def _xMeanCov(self, xX, xG0, xG1):
        '''
        Computes the mean and covariance between the latent variables.
        You can provide one or n latent variables.
        ----------------------------------------------------------------------------
        Input:
        xX              : [D] array of fixed effects or
                          [n*D] array of fixed effects for each latent variable
        xG0             : [k0] array of random effects or
                          [n*k0] array of random effects for each latent variable
        xG1             : [k1] array of random effects or
                          [n*k1] array of random effects for each latent variable
        -----------------------------------------------------------------------------
        Output:
        xmean           : [n*D] means of the provided latent variables
        xK01            : [N] or [n*N] covariance between provided and prior latent
                          variables
        xkk             : float or [n] covariance between provided latent variables 
        -----------------------------------------------------------------------------
        '''
        xmean = xX.dot(self._beta)

        if xG0 is not None:
            xK01 = self._sig02*(xG0.dot(self._G0.T))
        else:
            if len(xX.shape)==1:
                xK01 = NP.zeros(self._N)
            else:
                xK01 = NP.zeros((xX.shape[0], self._N))

        if len(xG0.shape)==1:
            xkk = self._sig02 * xG0.dot(xG0) + self._sign2
        else:
            xkk = self._sig02 * dotd(xG0,xG0.T) + self._sign2

        if xG1 is not None:
            xK01 += self._sig12*(xG1.dot(self._G1.T))
            if len(xG0.shape)==1:
                xkk += self._sig12 * xG1.dot(xG1)
            else:
                xkk += self._sig12 * dotd(xG1,xG1.T)

        return (xmean,xK01,xkk)

    def _predict(self, xmean, xK01, xkk, prob):
        pass

    def predict(self, xX, xG0, xG1=None, prob=True):
        '''
        Compute the probability of y=1.0, for each provided latent variable.
        It can instead return only the most probable class by setting prob=False.
        --------------------------------------------------------------------------
        Input:
        xX              : [D] array of fixed effects or
                          [n*D] array of fixed effects for each latent variable
        xG0             : [k0] array of random effects or
                          [n*k0] array of random effects for each latent variable
        xG1             : [k1] array of random effects or
                          [n*k1] array of random effects for each latent variable
        prob            : whether you want the probabilities or not.
        --------------------------------------------------------------------------
        Output:
        Array of probabilities or the most probable classes.
        --------------------------------------------------------------------------
        '''
        self._updateConstants()

        if len(xX.shape)==1:
            assert xX.shape[0]==self._D
            assert xG0.shape[0]==self._G0.shape[1]
            if xG1 is not None:
                assert xG1.shape[0]==self._G1.shape[1]

        elif len(xX.shape)==2:
            assert xX.shape[1]==self._D
            assert xG0.shape[1]==self._G0.shape[1]
            if xG1 is not None:
                assert xG1.shape[1]==self._G1.shape[1]

        else:
            assert False

        (xmean,xK01,xkk) = self._xMeanCov(xX, xG0, xG1)

        ps = self._predict(xmean, xK01, xkk, prob)
        if prob==False:
            return self._in2outy(ps)
        return ps

    def _in2outy(self, yin):
        '''
        Translates the labels used in this class (-1.0, 1.0) to the labels
        being used by the user.
        '''
        yout = NP.empty(yin.shape[0], dtype=self._outYType)
        yout[yin == -1] = self._mapIn2OutY[-1]
        yout[yin == +1] = self._mapIn2OutY[+1]
        return yout

    def _updateApproximation(self):
        pass

    def optimize(self, optSig02=True, optSig12=True, optSign2=True, optBeta=True):
        '''
        Minimize the cost, which is the negative of marginal loglikelihood plus the
        penalty, by adjusting the hyperparameters.
        --------------------------------------------------------------------------
        Input:
        optSig02, optSig12, optSign2, and optSigBeta can be True or False and
        are used to choose which hyperparameters are going to be optimized.
        --------------------------------------------------------------------------
        '''
        self._updateConstants()

        tol = 1e-6
        niterNoImprov = 5
        lowerBound = 1e-6

        betaScale = 4.0

        # initial values
        if optSig02:
            self.sig02 = 1.0
        if optSig12:
            self.sig12 = lowerBound
        if optSign2:
            self.sign2 = lowerBound
        if optBeta:
            self.beta = NP.zeros(self._D)
        
        bestCost = self._optimize(lowerBound, optSig02, optSig12, optSign2, optBeta)
        bestSolution = self._wrap_hyp(True, True, True, True)

        i = 0
        ntries = 0
        while i < niterNoImprov:
            if optSig02:
                self.sig02 = max(NP.random.chisquare(1), lowerBound)
            if optSig12:
                self.sig12 = max(NP.random.chisquare(1), lowerBound)
            if optSign2:
                self.sign2 = max(NP.random.chisquare(1), lowerBound)
            if optBeta:
                self.beta = NP.random.normal(scale=betaScale, size=self._D)

            cost = self._optimize(lowerBound, optSig02, optSig12, optSign2, optBeta)
            if cost < bestCost:
                bestCost = cost
                bestSolution = self._wrap_hyp(True, True, True, True)
                if abs(cost-bestCost) > tol:
                    i = 0
                else:
                    i += 1
            else:
                i += 1
            ntries += 1

        #if self._verbose:
        #    Pr.prin('Number of tries: {}.'.format(ntries))

        self._unwrap_hyp(bestSolution, True, True, True, True)

        # If K0 is null, marginal likelihood is independent on
        # self._sig02. Thus, let it be 0.0 by default.
        if not self._isK0Set:
            self.sig02 = 0.0

        # If K1 is null, marginal likelihood is independent on
        # self._sig12. Thus, let it be 0.0 by default.
        if not self._isK1Set:
            self.sig12 = 0.0

        self._check_sigmas_at_zero(lowerBound, optSig02, optSig12, optSign2, optBeta)
        
    def _optimize(self, lowerBound, optSig02=True, optSig12=True, optSign2=True, optBeta=True):
        def func(x):
            self._unwrap_hyp(x, optSig02, optSig12, optSign2, optBeta)
            return -self.marginal_loglikelihood()

        def grad(x):
            self._unwrap_hyp(x, optSig02, optSig12, optSign2, optBeta)
            return -self._mll_gradient()

        x = self._wrap_hyp(optSig02, optSig12, optSign2, optBeta)

        # bounds on sig02, sig12, sign2, and w
        bounds = []
        maxsig2 = 50.0
        maxbeta = 50.0
        if optSig02:
            bounds.append((lowerBound,maxsig2))
        if optSig12:
            bounds.append((lowerBound,maxsig2))
        if optSign2:
            bounds.append((lowerBound,maxsig2))
        if optBeta:
            bounds += [(-maxbeta,maxbeta)]*self._D

        (xfinal,aa,bb) = SP.optimize.fmin_tnc(func, x, fprime=grad, bounds=bounds, disp=0)
#        (xfinal,cost,msgs) = SP.optimize.fmin_l_bfgs_b(func, x, fprime=grad, bounds=bounds,
#                                                        disp=self._verbose)
#                                                        disp=False)
        self._unwrap_hyp(xfinal, optSig02, optSig12, optSign2, optBeta)
        marg = self.marginal_loglikelihood()
#        marg = -cost
        
        #if self._verbose:
        #    Pr.prin('Best hyperparameters. sig02 '+str(self._sig02)+' sig12 '+str(self._sig12)+' sign2 '+str(self._sign2)+' beta '+str(self._beta))
        #    Pr.prin('marg '+str(marg))

        return -marg

    def _check_sigmas_at_zero(self, lowerBound, optSig02, optSig12, optSign2, optBeta):
        pmargll = self.marginal_loglikelihood()

        psig02 = self._sig02
        if optSig02:
            if abs(self._sig02-lowerBound)<1e-7:
                self.sig02 = 0.0

        psig12 = self._sig12
        if optSig12:
            if abs(self._sig12-lowerBound)<1e-7:
                self.sig12 = 0.0

        psign2 = self._sign2
        if optSign2:
            if abs(self._sign2-lowerBound)<1e-7:
                self.sign2 = 0.0

        margll = self.marginal_loglikelihood()

        if margll < pmargll:
            self.sig02 = psig02
            self.sig12 = psig12
            self.sign2 = psign2

    def _is_kernel_zero(self):
        return self._sig02==0.0 and self._sig12==0.0 and self._sign2==0.0
    
    def marginal_loglikelihood(self):
        '''
        Returns (regular marginal loglikelihood) - penalty
        '''
        margll = self._regular_marginal_loglikelihood()

        if self._penalty is None:
            return margll

        if self._penalizeBias:
            assert self._hasBias, "You set to penalize bias but there isn't one."
            beta = self.beta
        else:
            beta = self._betaNoBias()

        if self._penalty == 'l1':
            assert False, 'Not implemented yet.'
            return margll - self._lambdaBeta * sum(abs(beta)) - self._lambdaSign2 * abs(self.sign2)

        elif self._penalty == 'l2':
            #return margll - self._lambdaBeta * NP.dot(beta,beta) - self._lambdaSign2 * self.sign2**2
            r = margll - self._lambdaBeta * NP.dot(beta,beta) - self._lambdaSign2 * self.sign2**2
            if self._isK0Set:
                r -= self._lambdaSig02 * self.sig02**2
            if self._isK1Set:
                r -= self._lambdaSig12 * self.sig12**2
            return r

        assert False, 'Unknown penalty.'

    def _betaNoBias(self):
        beta = self.beta.copy()
        if self._hasBias:
            assert self._hasBias, "You set to penalize bias but there isn't one."
            beta = NP.concatenate( (beta[:self._biasIndex], beta[self._biasIndex+1:]) )
        return beta

    def _mll_gradient(self, optSig02=True, optSig12=True, optSign2=True, optBeta=True):
        '''
        Marginal loglikelihood gradient.
        '''
        g = self._rmll_gradient()
        if self._penalty is None or optBeta is False:
            return g

        if optSig02:
            sig02Index = 0
        if optSig12:
            sig12Index = 0+optSig02
        if optSign2:
            sign2Index = optSig02 + optSig12

        firstIndex = optSig02 + optSig12 + optSign2
        
        gs = NP.zeros(g.shape[0])
        if self._penalty == 'l1':
            assert False, 'Not implemented yet.'

        elif self._penalty == 'l2':

            gs[firstIndex:] = 2.0 * self._lambdaBeta * self.beta

            if optSign2:
                gs[sign2Index] = 2.0 * self._lambdaSign2 * self.sign2

            if self._isK0Set and optSig02:
                gs[sig02Index] = 2.0 * self._lambdaSig02 * self.sig02
            if self._isK1Set and optSig12:
                gs[sig12Index] = 2.0 * self._lambdaSig12 * self.sig12
        else:
            assert False, 'Unknown penalty.'

        return g - gs

    def _wrap_hyp(self, optSig02, optSig12, optSign2, optBeta):
        x = NP.empty(sum([optSig02, optSig12, optSign2]) + sum([optBeta])*self._D)
        i = 0
        if optSig02:
            x[i] = self._sig02
            i += 1
        if optSig12:
            x[i] = self._sig12
            i += 1
        if optSign2:
            x[i] = self._sign2
            i += 1
        if optBeta:
            x[i:] = self._beta
        return x

    def _unwrap_hyp(self, x, optSig02, optSig12, optSign2, optBeta):
        i = 0
        if optSig02:
            self.sig02 = x[i]
            i += 1
        if optSig12:
            self.sig12 = x[i]
            i += 1
        if optSign2:
            self.sign2 = x[i]
            i += 1
        if optBeta:
            self.beta = x[i:]

    def _calculateMean(self):
        return self._X.dot(self._beta)

    def _updateConstants(self):
        pass

class GLMM_N3K1(GLMM):
    def __init__(self, penalty=None, penalizeBias=False, debug=False):
        GLMM.__init__(self, penalty=penalty, penalizeBias=penalizeBias, debug=debug)
        self._K0 = None
        self._K1 = None

    def setK(self, K0, K1=None):
        '''
        set the Kernels K0 and K1.
        --------------------------------------------------------------------------
        Input:
        K0 : [N*N] array, random effects covariance (positive definite)
        K1 : [N*N] array, random effects covariance (positive definite)(optional)
        --------------------------------------------------------------------------
        '''
        assert type(K0) is NP.ndarray, 'K0 must be a numpy.ndarray.'
        assert NP.all(NP.isfinite(K0) & ~NP.isnan(K0)), 'K0 must have only numbers.'
        assert len(K0.shape)==2, 'K0 must be a 2-dimensional array.'
        assert K0.shape[0] > 0
        assert K0.shape[1] > 0
        assert check_symmetry(K0), 'K0 must be a symmetric matrix.'
        assert check_definite_positiveness(K0), 'K0 must be a definite positive matrix.'
        self._K0 = K0

        if K1 is not None:
            assert type(K1) is NP.ndarray, 'K1 must be a numpy.ndarray.'
            assert NP.all(NP.isfinite(K1) & ~NP.isnan(K1)), 'K1 must have only numbers.'
            assert len(K1.shape)==2, 'K1 must be a 2-dimensional array.'
            assert K1.shape[0] > 0
            assert K1.shape[1] > 0
            assert check_symmetry(K1), 'K1 must be a symmetric matrix.'
            assert check_definite_positiveness(K1), 'K1 must be a definite positive matrix.'
            self._K1 = K1

        self._isK0Set = K0 is not None
        self._isK1Set = K1 is not None

        self._updateConstantsCount += 1

    def _updateConstants(self):
        '''
        Updates some constant members, which is needed when the user, for e.g., change
        the effects.
        '''
        if self._updateConstantsCount == 0:
            return

        assert self._X is not None, 'You must set X.'

        if self._G0 is not None and self._K0 is not None:
            assert abs(self._K0-self._G0.dot(self._G0.T)).max() < NP.sqrt(NP.finfo(NP.float).eps), 'You have set both G0 and K0, but K0!=G0*G0^T.'

        if self._G1 is not None and self._K1 is not None:
            assert abs(self._K1-self._G1.dot(self._G1.T)).max() < NP.sqrt(NP.finfo(NP.float).eps), 'You have set both G1 and K1, but K1!=G1*G1^T.'

        self._N = self._X.shape[0]
        self._D = self._X.shape[1]

        if self._K0 is None:
            if self._G0 is None:
                self._K0 = NP.zeros([self._N, self._N])
            else:
                self._K0 = self._G0.dot(self._G0.T)

        if self._K1 is None:
            if self._G1 is None:
                self._K1 = NP.zeros([self._N, self._N])
            else:
                self._K1 = self._G1.dot(self._G1.T)

        assert self._K0.shape[0]==self._X.shape[0], 'K0 (or G0) and X must have the same number of rows.'
        assert self._K1.shape[0]==self._X.shape[0], 'K1 (or G1) and X must have the same number of rows.'

        assert self._y is not None, 'You must set y.'
        assert self._y.shape[0]==self._X.shape[0], 'X and y have incompatible sizes.'
        
        self._updateApproximationCount += 1
        self._updateConstantsCount = 0

    # If A is dxN, we have O(dN^2) operations.
    def _ldotK(self, A):
        #TODO: pre-calculate in updateApproximationBegin
        R = self._sig02 * dot(A, self._K0) + A * self._sign2
        if self._K1 is not None:
            R += self._sig12 * dot(A, self._K1)
        return R

    # If A is Nxd, we have O(dN^2) operations.
    def _rdotK(self, A):
        #TODO: pre-calculate in updateApproximationBegin
        R = self._sig02 * dot(self._K0, A) + A * self._sign2
        if self._K1 is not None:
            R += self._sig12 * dot(self._K1, A)
        return R

    # Return K's diagonal, in O(N).
    def _dKn(self):
        #TODO: pre-calculate in updateApproximationBegin
        d = self._sig02 * NP.diag(self._K0) + self._sign2
        if self._K1 is not None:
            d += self._sig12 * NP.diag(self._K1)
        return d

    def _calculateLn(self, K, D):
        Bn = ddot(D, ddot(K, D, left=False), left=True)
        Bn[NP.diag_indices_from(Bn)] += 1.0
        Ln = cholesky(Bn, lower=True, check_finite=False)
        return Ln


class GLMM_N1K3(GLMM):
    def __init__(self, penalty=None, penalizeBias=False, debug=False):
        GLMM.__init__(self, penalty=penalty, penalizeBias=penalizeBias, debug=debug)

    # If A is dxN, we have O(kdN) operations (assuming N>=k>=d).
    def _ldotK(self, A):
        R = self._sig02 * (A.dot(self._G0).dot(self._G0.T)) + A * self._sign2
        if self._G1 is not None:
            R += self._sig12 * (A.dot(self._G1).dot(self._G1.T))
        return R

    # If A is Nxd, we have O(kdN) operations (assuming N>=k>=d).
    def _rdotK(self, A):
        R = self._sig02 * (self._G0.dot(self._G0.T.dot(A))) + A * self._sign2
        if self._G1 is not None:
            R += self._sig12 * (self._G1.dot(self._G1.T.dot(A)))
        return R

    # Return K's diagonal, in O((k0+k1)*N).
    def _dKn(self):
        d = self._sig02 * dotd(self._G0, self._G0.T) + self._sign2
        if self._G1 is not None:
            d += self._sig12 * dotd(self._G1, self._G1.T)
        return d

    def _updateConstants(self):
        '''
        Updates some constant members, which is needed when the user, for e.g., change
        the effects.
        '''
        if self._updateConstantsCount == 0:
            return

        assert self._X is not None, 'You must set X.'

        self._N = self._X.shape[0]
        self._D = self._X.shape[1]

        if self._G0 is None:
            self._G0 = NP.zeros([self._N, 1])

        if self._G1 is None:
            self._G1 = NP.zeros([self._N, 1])

        assert self._G0.shape[0]==self._X.shape[0], 'G0 and X must have the same number of rows.'
        assert self._G1.shape[0]==self._X.shape[0], 'G1 and X must have the same number of rows.'

        assert self._y is not None, 'You must set y.'
        assert self._y.shape[0]==self._X.shape[0], 'X and y have incompatible sizes.'

        self._updateApproximationCount += 1
        self._updateConstantsCount = 0

    def _calculateG01(self):
        G01 = NP.concatenate( (NP.sqrt(self._sig02)*self._G0, NP.sqrt(self._sig12)*self._G1), axis=1 )
        return G01

    def _calculateLk(self, G01, D):
        Bk  = dot(G01.T, ddot(D, G01, left=True))
        Bk[NP.diag_indices_from(Bk)] += 1.0
        Lk = cholesky(Bk, lower=True, check_finite=False)
        return Lk
