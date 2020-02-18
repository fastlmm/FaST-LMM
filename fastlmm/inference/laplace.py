from __future__ import absolute_import
from fastlmm import Pr
import scipy as sp
import numpy as NP
from numpy import dot
import scipy.integrate
from scipy.linalg import cholesky,solve_triangular
from fastlmm.external.util.math import check_definite_positiveness,check_symmetry,mvnormpdf,ddot,trace2,dotd
from fastlmm.external.util.math import stl, stu
from fastlmm.inference.glmm import GLMM_N1K3, GLMM_N3K1
from fastlmm.inference.likelihood import LogitLikelihood, ProbitLikelihood
from fastlmm import Pr
import sys
from six.moves import range

'''
    Important! Always run test.py in the current folder for unit testing after
    changes have been made.
'''

class DebugUACall(object):
    def __init__(self, innerIters, lastGrad, success, beta, sig02, sig12, sign2):
        self.innerIters = innerIters
        self.lastGrad = lastGrad
        self.success = success
        self.beta = beta
        self.sig02 = sig02
        self.sig12 = sig12
        self.sign2 = sign2

class LaplaceGLMM(object):
    def __init__(self, link):
        self._lasta = None
        self._debugUACalls = []
        self._link = link
        if link == "logistic":
            self._likelihood = LogitLikelihood()
        elif link == "erf":
            self._likelihood = ProbitLikelihood()
        else:
            assert False, "Unknown link function."

    def _calculateW(self, f):
        W = -self._likelihood.hessian_log(f)
        return W

    def _lineSearch(self, a, aprev, m):
        da = a - aprev
        def fobj(alpha):
            a = aprev + alpha*da
            f = self._rdotK(a) + m
            return -(self._likelihood.log(f, self._y) - (f-m).dot(a)/2.0)

        (alpha,obj,iter,funcalls) = sp.optimize.brent(fobj, brack=(0.0,1.0), full_output=True, tol=1e-4, maxiter=10)
        obj = -obj
        a = aprev + alpha*da
        f = self._rdotK(a) + m
        return (f, a, obj)

    def _calculateUAGrad(self, f, a):
        grad = self._likelihood.gradient_log(f, self._y) - a
        return grad

    def printDebug(self):
        assert self._debug is True
        from tabulate import tabulate
        iters = [self._debugUACalls[i].innerIters for i in range(len(self._debugUACalls))]
        sig02 = [self._debugUACalls[i].sig02 for i in range(len(self._debugUACalls))]
        sig12 = [self._debugUACalls[i].sig12 for i in range(len(self._debugUACalls))]
        sign2 = [self._debugUACalls[i].sign2 for i in range(len(self._debugUACalls))]
        gradMeans = [NP.mean(abs(self._debugUACalls[i].lastGrad)) for i in range(len(self._debugUACalls))]

        Pr.prin("*** Update approximation ***")
        Pr.prin("calls: %d" % (len(self._debugUACalls),))

        table = [["", "min", "max", "mean"],
            ["iters", min(iters), max(iters), NP.mean(iters)],
            ["|grad|_{mean}", min(gradMeans), max(gradMeans), NP.mean(gradMeans)],
            ["sig01", min(sig02), max(sig02), NP.mean(sig02)],
            ["sig11", min(sig12), max(sig12), NP.mean(sig12)],
            ["sign1", min(sign2), max(sign2), NP.mean(sign2)]]
        Pr.prin(tabulate(table))

    def _updateApproximation(self):
        '''
        Calculates the Laplace approximation for the posterior.
        It can be defined by two variables: f mode and W at f mode.
        '''
        if self._updateApproximationCount == 0:
            return

        if self._is_kernel_zero():
            self._updateApproximationCount = 0
            return

        self._updateApproximationBegin()

        gradEpsStop = 1e-10
        objEpsStop = 1e-8
        gradEpsErr = 1e-3

        self._mean = self._calculateMean()
        m = self._mean

        if self._lasta is None or self._lasta.shape[0] != self._N:
            aprev = NP.zeros(self._N)
        else:
            aprev = self._lasta

        fprev = self._rdotK(aprev) + m
        objprev = self._likelihood.log(fprev, self._y) - (fprev-m).dot(aprev)/2.0
        ii = 0
        line_search = False
        maxIter = 1000
        failed = False
        failedMsg = ''
        while ii < maxIter:
            grad = self._calculateUAGrad(fprev, aprev)
            if NP.mean(abs(grad)) < gradEpsStop:
                a = aprev
                f = fprev
                break

            # The following is just a Newton step (eq. (3.18) [1]) to maximize
            # log(p(F|X,y)) over F
            g = self._likelihood.gradient_log(fprev, self._y)
            W = self._calculateW(fprev)
            b = W*(fprev-m) + g

            a = self._calculateUAa(b, W)

            if line_search:
                (f, a, obj) = self._lineSearch(a, aprev, m)
            else:
                f = self._rdotK(a) + m
                obj = self._likelihood.log(f, self._y) - (f-m).dot(a)/2.0

            if abs(objprev-obj) < objEpsStop :
                grad = self._calculateUAGrad(f, a)
                break
            if obj > objprev:
                fprev = f
                objprev = obj
                aprev = a
            else:
                if line_search:
                    grad = self._calculateUAGrad(fprev, aprev)
                    a = aprev
                    f = fprev
                    break
                line_search = True
            ii+=1

        self._lasta = a

        err = NP.mean(abs(grad))
        if err > gradEpsErr:
            failed = True
            failedMsg = 'Gradient not too small in the Laplace update approximation.\n'
            failedMsg = failedMsg+"Problem in the f mode estimation. |grad|_{mean} = %.6f." % (err,)

        if ii>=maxIter:
            failed = True
            failedMsg = 'Laplace update approximation did not converge in less than maxIter.'

        if self._debug:
            self._debugUACalls.append(self.DebugUACall(
                ii, grad, not failed, self.beta, self._sig02, self._sig12, self._sign2))

        if failed:
            Pr.prin('Laplace update approximation failed. The failure message is the following.')
            Pr.prin(failedMsg)
            sys.exit('Stopping program.')

        self._updateApproximationEnd(f, a)

        self._updateApproximationCount = 0

    def _predict(self, meanstar, kstar, kstarstar, prob):
        self._updateConstants()
        self._updateApproximation()

        if NP.isscalar(kstarstar):
            return self._predict_each(meanstar, kstar, kstarstar, prob)

        n = len(kstarstar)
        ps = NP.zeros(n)

        for i in range(n):
            ps[i] = self._predict_each(meanstar[i], kstar[i,:], kstarstar[i], prob)

        return ps

class LaplaceGLMM_N1K3(GLMM_N1K3, LaplaceGLMM):
    def __init__(self, link, penalty=None, penalizeBias=False, debug=False):
        GLMM_N1K3.__init__(self, penalty=penalty, penalizeBias=penalizeBias, debug=debug)
        LaplaceGLMM.__init__(self, link)
        self._link = link

    def _updateApproximationBegin(self):
        self._G01 = self._calculateG01()

    def _calculateUAa(self, b, W):
        A = 1.0 + W*self._sign2
        V = W/A
        Lk = self._calculateLk(self._G01, V)

        Gtb = dot(self._G01.T, b)
        GtV = ddot(self._G01.T, V, left=False)
        LtLGtV = stu(Lk.T, stl(Lk, GtV))
        LtLGtVG = dot(LtLGtV, self._G01)
        bn = self._sign2*b
        a = b + dot(dot(GtV.T, LtLGtVG) - GtV.T, Gtb)\
              + dot(GtV.T, dot(LtLGtV, bn)) - V*bn

        return a

    def _updateApproximationEnd(self, f, a):
        self._f = f
        self._a = a

        self._W = self._calculateW(f)
        self._Wsq = NP.sqrt(self._W)

        self._A = 1.0 + self._W * self._sign2
        self._V = self._W/self._A
        self._Lk = self._calculateLk(self._G01, self._V)

    def _updateApproximation(self):
        LaplaceGLMM._updateApproximation(self)

    def _regular_marginal_loglikelihood(self):
        self._updateConstants()
        self._updateApproximation()

        if self._is_kernel_zero():
            return self._likelihood.log(self._mean, self._y)

        (f,a) = (self._f,self._a)

        loglike = self._likelihood.log(f, self._y)

        r = loglike - dot(f-self._mean,a)/2.0 - sum(NP.log(NP.diag(self._Lk)))\
            - sum(NP.log( self._A ))/2.0
        assert NP.isfinite(r), 'Not finite regular marginal loglikelihood.'

        return r

    def _rmll_gradient(self, optSig02=True, optSig12=True, optSign2=True, optBeta=True):
        self._updateConstants()
        self._updateApproximation()

        (f,a)=(self._f,self._a)
        (W,Wsq) = (self._W,self._Wsq)
        Lk = self._Lk

        m = self._mean
        X = self._X
        G0 = self._G0
        G1 = self._G1
        sign2 = self._sign2
        G01 = self._G01

        #g = self._likelihood.gradient_log(f)
        #a==g

        h = self._likelihood.third_derivative_log(f)

        V = W/self._A

        d = self._dKn()
        G01tV = ddot(G01.T, V, left=False)
        H = stl(Lk, G01tV)
        dkH = self._ldotK(H)
        diags = (d - sign2**2 * V - dotd(G01, dot(dot(G01tV, G01), G01.T))\
            - 2.0*sign2*dotd(G01, G01tV) + dotd(dkH.T, dkH)) * h

        ret = []

        if optSig02:
            dK0a = dot(G0, dot(G0.T, a))
            t = V*dK0a - dot(H.T, dot(H, dK0a))
            dF0 = dK0a - self._rdotK(t)

            LkG01VG0 = dot(H, G0)
            VG0 = ddot(V, G0, left=True)

            ret0 = dot(a, dF0) - 0.5*dot(a, dK0a) + dot(f-m, t)\
                + 0.5*NP.sum( diags*dF0 )\
                + -0.5*trace2(VG0, G0.T) + 0.5*trace2( LkG01VG0.T, LkG01VG0 )

            ret.append(ret0)

        if optSig12:
            dK1a = dot(G1, dot(G1.T, a))
            t = V*dK1a - dot(H.T, dot(H, dK1a))
            dF1 = dK1a - self._rdotK(t)

            LkG01VG1 = dot(H, G1)
            VG1 = ddot(V, G1, left=True)

            ret1 = dot(a, dF1)- 0.5*dot(a, dK1a) + dot(f-m, t)\
                + 0.5*NP.sum( diags*dF1 )\
                + -0.5*trace2(VG1, G1.T) + 0.5*trace2( LkG01VG1.T, LkG01VG1 )

            ret.append(ret1)

        if optSign2:
            t = V*a - dot(H.T, dot(H, a))
            dFn = a - self._rdotK(t)

            retn = dot(a, dFn)- 0.5*dot(a, a) + dot(f-m, t)\
                + 0.5*NP.sum( diags*dFn )\
                + -0.5*NP.sum(V) + 0.5*trace2( H.T, H )

            ret.append(retn)

        if optBeta:
            t = ddot(V, X, left=True) - dot(H.T, dot(H, X))
            dFbeta = X - self._rdotK(t)

            retbeta = dot(a, dFbeta) + dot(f-m, t)
            for i in range(dFbeta.shape[1]):
                retbeta[i] += 0.5*NP.sum( diags*dFbeta[:,i] )

            ret.extend(retbeta)

        ret = NP.array(ret)
        assert NP.all(NP.isfinite(ret)), 'Not finite regular marginal loglikelihood gradient.'

        return ret

    def _predict(self, meanstar, kstar, kstarstar, prob):
        return LaplaceGLMM._predict(self, meanstar, kstar, kstarstar, prob)

    def _predict_each(self, meanstar, kstar, xstarstar, prob):
        '''
        Calculates the probability of being 1, or the most probable label
        if prob=False.
        --------------------------------------------------------------------------
        Input:
        meanstar        : input mean.
        kstar           : covariance between provided and prior latent variables.
        xstarstar       : variance of the latent variable.
        prob            : True for probability calculation or False for returning
                          the most probable label.
        '''
        a = self._a

        fstarmean = meanstar + kstar.dot(a)

        if prob is False:
            if fstarmean > 0.0:
                return +1.0
            return -1.0

        r0Tr0 = kstar.dot(self._V*kstar)

        r1 = stl(self._Lk, dot(self._G01.T, self._V*kstar))

        fstarvar = xstarstar - r0Tr0 + r1.dot(r1)

        return self._likelihood.intOverGauss(fstarmean, fstarvar)

class LaplaceGLMM_N3K1(GLMM_N3K1, LaplaceGLMM):
    def __init__(self, link, penalty=None, penalizeBias=False, debug=False):
        GLMM_N3K1.__init__(self, penalty=penalty, penalizeBias=penalizeBias, debug=debug)
        LaplaceGLMM.__init__(self, link)
        self._link = link

    def _updateApproximationBegin(self):
        self._K = NP.eye(self._N) * self._sign2
        if self._isK0Set:
            self._K += self._sig02*(self._K0 if self._K0 is not None else dot(self._G0, self._G0.T))
        if self._isK1Set:
            self._K += self._sig12*(self._K1 if self._K1 is not None else dot(self._G1, self._G1.T))

    def _calculateUAa(self, b, W):
        Wsq = NP.sqrt(W)
        Ln = self._calculateLn(self._K, Wsq)
        a = b - Wsq * stu(Ln.T, stl(Ln, Wsq*dot(self._K,b)))
        return a

    def _updateApproximationEnd(self, f, a):
        self._f = f
        self._a = a

        self._W = self._calculateW(f)
        self._Wsq = NP.sqrt(self._W)

        self._A = 1.0 + self._W * self._sign2
        V = self._W/self._A
        self._Ln = self._calculateLn(self._K, self._Wsq)

    def _regular_marginal_loglikelihood(self):
        self._updateConstants()
        self._updateApproximation()

        if self._is_kernel_zero():
            return self._likelihood.log(self._mean, self._y)

        (f,a) = (self._f,self._a)

        loglike = self._likelihood.log(f, self._y)

        r = loglike - dot(f-self._mean,a)/2.0 - sum(NP.log(NP.diag(self._Ln)))
        assert NP.isfinite(r), 'Not finite regular marginal loglikelihood.'

        return r

    def _rmll_gradient(self, optSig02=True, optSig12=True, optSign2=True, optBeta=True):
        self._updateConstants()
        self._updateApproximation()

        W = self._W
        Wsq = self._Wsq
        f = self._f
        K = self._K
        K0 = self._K0
        K1 = self._K1
        m = self._mean
        a = self._a
        Ln = self._Ln
        X = self._X

        LnWsq = stl(Ln, NP.diag(Wsq))
        LnWsqK = dot(LnWsq, K)

        d = self._dKn()
        h = self._likelihood.third_derivative_log(f)
        diags = (d - dotd(LnWsqK.T, LnWsqK)) * h

        ret = []

        if optSig02:
            dK0a = dot(K0, a)
            dF0 = dK0a - dot(LnWsqK.T, dot(LnWsq, dK0a))

            r = dot(a, dF0) - dot(a, dF0) + 0.5*dot(a, dK0a)\
                + 0.5*NP.sum( diags*dF0 )\
                - 0.5*trace2( LnWsq.T, dot(LnWsq,K0) )

            ret.append(r)

        if optSig12:
            dK1a = dot(K1, a)
            dF1 = dK1a - dot(LnWsqK.T, dot(LnWsq, dK1a))

            r = dot(a, dF1) - dot(a, dF1) + 0.5*dot(a, dK1a)\
                + 0.5*NP.sum( diags*dF1 )\
                - 0.5*trace2( LnWsq.T, dot(LnWsq,K1) )

            ret.append(r)

        if optSign2:
            dFn = a - dot(LnWsqK.T, dot(LnWsq, a))

            r = dot(a, dFn) - dot(a, dFn) + 0.5*dot(a, a)\
                + 0.5*NP.sum( diags*dFn )\
                - 0.5*trace2( LnWsq.T, LnWsq )

            ret.append(r)

        if optBeta:
            dFmb = -dot(LnWsqK.T, dot(LnWsq, X))
            dFb = dFmb+X

            r = dot(a, dFb) - dot(a, dFmb)\
                + 0.5*NP.sum( diags*dFb.T, 1)

            ret += list(r)

        ret = NP.array(ret)
        assert NP.all(NP.isfinite(ret)), 'Not finite regular marginal loglikelihood gradient.'

        return ret

    def _updateApproximation(self):
        LaplaceGLMM._updateApproximation(self)

    def _predict(self, meanstar, kstar, kstarstar, prob):
        return LaplaceGLMM._predict(self, meanstar, kstar, kstarstar, prob)

    def _predict_each(self, meanstar, kstar, xstarstar, prob):
        '''
        Calculates the probability of being 1, or the most probable label
        if prob=False.
        --------------------------------------------------------------------------
        Input:
        meanstar        : input mean.
        kstar           : covariance between provided and prior latent variables.
        xstarstar       : variance of the latent variable.
        prob            : True for probability calculation or False for returning
                          the most probable label.
        '''

        fstarmean = meanstar + kstar.dot(self._a)

        if prob is False:
            if fstarmean > 0.0:
                return +1.0
            return -1.0

        r = stl(self._Ln, self._Wsq*kstar)

        fstarvar = xstarstar - dot(r,r)

        return self._likelihood.intOverGauss(fstarmean, fstarvar)
