from __future__ import absolute_import
import scipy.stats as ST
import numpy as NP

logit_sigmoid = lambda x: 1.0 / (1.0 + NP.exp(-x))
probit_sigmoid = lambda x: ST.norm.cdf(x)

class Likelihood:
    def __init__(self):
        pass

    # likelihood
    def plain(self, f, y):
        raise NotImplementedError
    
    # log likelihood
    def log(self, f, y):
        raise NotImplementedError
    
    # gradient of the log likelihood
    def gradient_log(self, f, y):
        raise NotImplementedError
    
    # hessian of the log likelihood
    def hessian_log(self, f):
        raise NotImplementedError

    def third_derivative_log(self, f):
        raise NotImplementedError

class ProbitLikelihood(Likelihood):
    def __init__(self):
        Likelihood.__init__(self)

    # likelihood
    def plain(self, f, y):
        return ST.norm._cdf(y*f).prod()

    # log likelihood
    def log(self, f, y):
        return ST.norm._logcdf(y*f).sum()

# Implements p(y|F) = prod_{i=1}^n p(y_i|f_i)
#                   = prod_{i=1}^n logistic(y_i*f_i).
class LogitLikelihood(Likelihood):
    def __init__(self):
        Likelihood.__init__(self)
        #self.sigmoid = lambda x: 1.0 / (1.0 + NP.exp(-x))
        self.sigmoid = logit_sigmoid 

    # likelihood
    def plain(self, f, y):
        return self.sigmoid(y*f).prod()
    
    # log likelihood
    def log(self, f, y): 
        yf = y*f 
        r = self.sigmoid(yf) 
        ok = r>0.0 
        r[ok] = NP.log( r[ok] ) 
        r[~ok] = yf[~ok]
        return r.sum()
    
    # gradient of the log likelihood
    def gradient_log(self, f, y):
        return y*self.sigmoid(-y*f)
    
    # hessian of the log likelihood
    def hessian_log(self, f):
        r = self.sigmoid(f)*self.sigmoid(-f)
        ok = r >= 1e-16
        r[~ok] = 1e-16
        return -r
        # maybe faster, but maybe less precise version
        # pi = self.sigmoid(f)
        # return -pi*(1.0-pi)

    def third_derivative_log(self, f):
        pi = self.sigmoid(f)
        return -pi*self.sigmoid(-f)*(1.0-2.0*pi)
        # maybe faster, but maybe less precise version
        # return -pi*(1.0-pi)*(1.0-2.0*pi)

    def intOverGauss(self, mu, sig2):
        # horta201306 we approximate \int logistic(x) N(x, mu, sig2) dx
        # by observing that logistic(x) is cdf of the standard logistic distribution
        # and that this distribution is similar in shape with the normal one
        # so we approximate the logistic pdf by a mixture of five normal
        # distributions, and then calculate the integral
        # we adopted the coefficients c and lambd from Rasmussen and Williams' GP toolbox
        # for Matlab.

        c = NP.array([1.146480988574439e+02,-1.508871030070582e+03,2.676085036831241e+03,-1.356294962039222e+03,7.543285642111850e+01])
        lambd = NP.array([0.44,0.41,0.40,0.39,0.36])

        sigs = 1.0/(lambd*NP.sqrt(2))

        z = mu / (  sigs * NP.sqrt(1.0 + sig2/sigs**2)  )

        return NP.sum(c*ST.norm.cdf(z))
