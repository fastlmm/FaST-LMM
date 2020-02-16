from __future__ import absolute_import
import numpy as NP
import scipy as SP
from numpy import dot
import scipy.stats as ST
from scipy.linalg import cholesky,solve_triangular
from fastlmm.external.util.math import check_definite_positiveness,check_symmetry,mvnormpdf,ddot,trace2
from fastlmm.external.util.math import stl, stu, dotd
from fastlmm.inference.glmm import GLMM_N1K3, GLMM_N3K1
from fastlmm.inference.likelihood import LogitLikelihood, ProbitLikelihood
from fastlmm import Pr
from . import likelihood as LH


'''
    Important! Always run test.py in the current folder for unit testing after
    changes have been made.
'''

class EPGLMM(object):
    def __init__(self, link):
        self._debugUACalls = []
        self._link = link
        if link == "logistic":
            self._likelihood = LogitLikelihood()
        elif link == "erf":
            self._likelihood = ProbitLikelihood()
        else:
            assert False, "Unknown link function."

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

        self._mean = self._calculateMean()
        m = self._mean

        ttau = NP.zeros(self._N)
        tnu = NP.zeros(self._N)
        sig2_ = NP.zeros(self._N)
        mu_ = NP.zeros(self._N)

        prevsig2 = self._dKn()

        prevmu = m.copy()

        converged = False
        outeriter = 1
        iterMax = 1000

        while outeriter <= iterMax and not converged:

            tau_ = 1.0/prevsig2 - ttau
            nu_ = prevmu/prevsig2 - tnu

            tt = tau_**2 + tau_
            stt = NP.sqrt(tt)
            c = (self._y*nu_) / stt

            # dan: using _cdf and _pdf instead of cdf and pdf I avoid
            # a lot of overhead due to error checking and other things
            nc_hz = NP.exp( ST.norm._logpdf(c) - ST.norm._logcdf(c) )

            hmu = nu_/tau_ + nc_hz*self._y/stt

            hsig2 = 1.0/tau_ - (nc_hz/tt) * (c + nc_hz)

            tempttau = 1.0/hsig2 - tau_
            ok = NP.bitwise_and(~NP.isnan(tempttau), abs(tempttau)>1e-8)
            ttau[ok] = tempttau[ok]

            tnu[ok] = hmu[ok]/hsig2[ok] - nu_[ok]

            (sig2, mu) = self._calculateSig2Mu(ttau, tnu, m)

            lastMuDiff = abs(mu-prevmu).max()
            lastSig2Diff = abs(sig2-prevsig2).max()
            if lastMuDiff  < 1e-6 and lastSig2Diff < 1e-6:
                converged = True

            prevmu = mu
            prevsig2 = sig2

            outeriter += 1

        if outeriter > iterMax:
            Pr.prin('EP did not converge. Printing debug information...')
            Pr.prin('beta ' + str(self.beta))
            Pr.prin('sig02 '+str(self.sig02)+' sig12 '+str(self.sig12)+' sign2 '+str(self.sign2))
            Pr.prin('abs(mu-prevmu).max() '+str(lastMuDiff))
            Pr.prin('abs(sig2-prevsig2).max() '+str(lastSig2Diff))

        self._updateApproximationEnd(ttau, tnu, tau_, nu_)
        self._updateApproximationCount = 0

class EPGLMM_N1K3(GLMM_N1K3, EPGLMM):
    def __init__(self, link, penalty=None, penalizeBias=False, debug=False):
        GLMM_N1K3.__init__(self, penalty=penalty, penalizeBias=penalizeBias, debug=debug)
        EPGLMM.__init__(self, link)
        self._link = link

    def _updateApproximationBegin(self):
        self._G01 = self._calculateG01()

    def _updateApproximationEnd(self, ttau, tnu, tau_, nu_):
        self._ttau = ttau
        self._tnu = tnu
        self._tau_ = tau_
        self._nu_ = nu_
        self._V = self._calculateV(ttau, self._sign2)
        self._Lk = self._calculateLk(self._G01, self._V)
        self._H = stl(self._Lk, ddot(self._G01.T, self._V, left=False))

    def _regular_marginal_loglikelihood(self):
        self._updateConstants()
        self._updateApproximation()

        if self._is_kernel_zero():
            return self._likelihood.log(self._mean, self._y)

        m = self._mean
        sign2 = self._sign2
        tnu = self._tnu
        ttau = self._ttau
        tau_ = self._tau_
        nu_ = self._nu_
        G01 = self._G01
        V = self._V
        Lk = self._Lk
        H = self._H

        A = ttau*sign2 + 1.0
        Ktnu = self._rdotK(tnu)
        HKtnu = dot(H,Ktnu)
        Hm = dot(H,m)
        ts_1 = 1.0/(tau_+ttau)
        mV = m*V

        r = 0.5*NP.sum(NP.log(1.0+ttau/tau_)) - 0.5*NP.sum(NP.log(A)) - NP.sum(NP.log(NP.diag(Lk)))\
            + NP.sum( ST.norm._logcdf((self._y*nu_)/NP.sqrt(tau_**2+tau_)) )\
            + 0.5*(dot(tnu, Ktnu) - dot(Ktnu.T, V*Ktnu) + dot(HKtnu.T, HKtnu) - dot(tnu, ts_1*tnu))\
            + dot(m,tnu) - dot(mV, Ktnu) + dot(Hm.T, HKtnu) - 0.5*(dot(mV, m) - dot(Hm.T, Hm))\
            + 0.5*dot(nu_*ts_1, (ttau/tau_)*nu_ - 2.0*tnu)

        return r

    def _rmll_gradient(self, optSig02=True, optSig12=True, optSign2=True, optBeta=True):
        self._updateConstants()
        self._updateApproximation()

        m = self._mean
        H = self._H
        V = self._V
        ttau = self._ttau
        tnu = self._tnu
        G0 = self._G0
        G1 = self._G1

        Smtnu = ttau*m - tnu
        KSmtnu = self._rdotK(Smtnu)

        b = Smtnu - V*KSmtnu + dot(H.T, dot(H, KSmtnu))

        ret = []
        if optSig02:
            r = 0.5*(dot(b, dot(G0, dot(G0.T, b))) - trace2(ddot(V, G0, left=True), G0.T)\
                + trace2(H.T, dot(dot(H, G0), G0.T)))

            ret.append(r)

        if optSig12:
            r = 0.5*(dot(b, dot(G1, dot(G1.T, b))) - trace2(ddot(V, G1, left=True), G1.T)\
                + trace2(H.T, dot(dot(H, G1), G1.T)))

            ret.append(r)

        if optSign2:
            r = 0.5*(dot(b, b) - NP.sum(V)\
                + trace2(H.T, H))

            ret.append(r)

        if optBeta:
            ret += list(-dot(b, self._X))

        return NP.array(ret)

    def _updateApproximation(self):
        EPGLMM._updateApproximation(self)

    def _calculateV(self, ttau, sign2):
        V = ttau/(ttau*sign2 + 1.0)
        return V

    def _calculateSig2Mu(self, ttau, tnu, mean):
        sign2 = self._sign2

        V = self._calculateV(ttau, sign2)
        G01 = self._G01

        Lk = self._calculateLk(G01, V)

        G01tV = ddot(G01.T, V, left=False)
        H = stl(Lk, G01tV)

        HK = self._ldotK(H)

        sig2 = self._dKn() - sign2**2*V - dotd(G01, dot(dot(G01tV, G01), G01.T))\
            - 2.0*sign2*dotd(G01, G01tV) + dotd(HK.T, HK)

        assert NP.all(NP.isfinite(sig2)), 'sig2 should be finite.'

        u = self._mean + self._rdotK(tnu)

        mu = u - self._rdotK(V*u) + self._rdotK(H.T.dot(H.dot(u)))
        assert NP.all(NP.isfinite(mu)), 'mu should be finite.'

        return (sig2, mu)

    def _predict(self, meanstar, kstar, kstarstar, prob):
        self._updateConstants()
        self._updateApproximation()

        m = self._mean
        tnu = self._tnu
        Lk = self._Lk
        H = self._H
        V = self._V

        Ktnu = self._rdotK(tnu)
        mKtnu = m + Ktnu
        Vkstar = ddot(V, kstar, left=True)
        Hkstar = dot(H, kstar)

        mustar = meanstar + dot(kstar.T,tnu) - dot(Vkstar.T, mKtnu) + dot(Hkstar.T, dot(H, mKtnu))

        if prob is False:
            if nom > 0.0:
                return +1.0
            return -1.0

        sig2star = kstarstar - dotd(kstar.T,Vkstar)\
            + dotd(dot(Hkstar.T, H), kstar) + self._sign2

        return LH.probit_sigmoid(mustar/NP.sqrt(1.0 + sig2star))

class EPGLMM_N3K1(GLMM_N3K1, EPGLMM):
    def __init__(self, link, penalty=None, penalizeBias=False, debug=False):
        GLMM_N3K1.__init__(self, penalty=penalty, penalizeBias=penalizeBias, debug=debug)
        EPGLMM.__init__(self, link)
        self._link = link

    def _updateApproximationBegin(self):
        self._K = NP.eye(self._N) * self._sign2
        if self._isK0Set:
            self._K += self._sig02*(self._K0 if self._K0 is not None else dot(self._G0, self._G0.T))
        if self._isK1Set:
            self._K += self._sig12*(self._K1 if self._K1 is not None else dot(self._G1, self._G1.T))

    def _regular_marginal_loglikelihood(self):
        self._updateConstants()
        self._updateApproximation()

        if self._is_kernel_zero():
            return self._likelihood.log(self._mean, self._y)

        m = self._mean
        K = self._K
        Ln = self._Ln
        ttau = self._ttau
        tnu = self._tnu
        tau_ = self._tau_
        nu_ = self._nu_
        LnSsq = self._LnSsq
        LnSsqK = self._LnSsqK

        LnSsqKtnu = dot(LnSsqK, tnu)
        ts_1 = 1.0/(tau_ + ttau)
        LnSsqm = dot(LnSsq, m)

        r = 0.5*NP.sum(NP.log(1.0 + ttau/tau_)) - NP.sum(NP.log(NP.diag(Ln)))\
            + NP.sum( ST.norm._logcdf( (self._y*nu_) / NP.sqrt(tau_**2+tau_) ) )\
            + 0.5*(dot(tnu, dot(K, tnu)) - dot(LnSsqKtnu,LnSsqKtnu) - dot(tnu*ts_1, tnu))\
            + dot(m, tnu) - dot(LnSsqm.T, LnSsqKtnu) - 0.5*dot(LnSsqm.T, LnSsqm)\
            + 0.5*dot(nu_*ts_1, (ttau/tau_)*nu_ - 2.0*tnu)

        return r

    def _rmll_gradient(self, optSig02=True, optSig12=True, optSign2=True, optBeta=True):
        self._updateConstants()
        self._updateApproximation()

        K0 = self._K0
        K1 = self._K1
        m = self._mean
        ttau = self._ttau
        tnu = self._tnu
        Ssq = self._Ssq
        LnSsq = self._LnSsq
        LnSsqK = self._LnSsqK

        Smtnu = ttau*m - tnu

        b = Smtnu - dot(LnSsq.T, dot(LnSsqK, Smtnu) )

        ret = []

        if optSig02:
            r = 0.5*(dot(b,dot(K0,b)) - trace2(LnSsq.T, dot(LnSsq, K0)))
            ret.append(r)

        if optSig12:
            r = 0.5*(dot(b,dot(K1,b)) - trace2(LnSsq.T, dot(LnSsq, K1)))
            ret.append(r)

        if optSign2:
            r = 0.5*(dot(b,b) - trace2(LnSsq.T, LnSsq))
            ret.append(r)

        if optBeta:
            r = -dot(b, self._X)
            ret += list(r)

        ret = NP.array(ret)
        assert NP.all(NP.isfinite(ret)), 'Not finite regular marginal loglikelihood gradient.'

        return ret

    def _updateApproximation(self):
        EPGLMM._updateApproximation(self)

    def _updateApproximationEnd(self, ttau, tnu, tau_, nu_):
        self._ttau = ttau
        self._tnu = tnu
        self._tau_ = tau_
        self._nu_ = nu_
        self._Ssq = NP.sqrt(ttau)
        self._Ln = self._calculateLn(self._K, self._Ssq)
        self._LnSsq = stl(self._Ln, NP.diag(self._Ssq))
        self._LnSsqK = dot(self._LnSsq, self._K)

    def _calculateSig2Mu(self, ttau, tnu, mean):
        Ssq = NP.sqrt(ttau)

        Ln = self._calculateLn(self._K, Ssq)

        LnSsq = stl(Ln, NP.diag(Ssq))
        V = dot(LnSsq,self._K)
        sig2 = self._dKn() - dotd(V.T,V)
        mu = mean + dot(self._K,tnu) - dot(V.T,dot(LnSsq,mean)) - dot(V.T,dot(V,tnu))

        return (sig2, mu)

    def _predict(self, meanstar, kstar, kstarstar, prob):
        self._updateConstants()
        self._updateApproximation()

        m = self._mean
        tnu = self._tnu
        K = self._K
        Ln = self._Ln
        Ssq = self._Ssq
        LnSsq = self._LnSsq
        LnSsqkstar = dot(LnSsq, kstar)
        LnSsqK = self._LnSsqK

        mustar = meanstar + dot(kstar.T,tnu) - dot(LnSsqkstar.T, dot(LnSsq,m) + dot(LnSsqK,tnu))

        if prob is False:
            if nom > 0.0:
                return +1.0
            return -1.0

        sig2star = kstarstar - dotd(dot(LnSsqkstar.T, LnSsq), kstar) + self._sign2

        return LH.probit_sigmoid(mustar/NP.sqrt(1.0 + sig2star))
