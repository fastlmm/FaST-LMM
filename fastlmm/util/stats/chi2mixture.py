import scipy as sp
import scipy.stats as st
import scipy.special
import fastlmm.util.mingrid as mingrid
import pdb
import logging

class chi2mixture(object):
    '''
    mixture here denotes the weight on the non-zero dof compnent
    '''
    __slots__ = ['scale','dof','mixture','imax','lrt','scalemin','scalemax',
                 'dofmin','dofmax', 'qmax', 'tol', 'isortlrt','qnulllrtsort',
                 'lrtsort','alteqnull','abserr','fitdof']

    def __init__(self, lrt, tol = 0.0, scalemin = 0.1, scalemax = 5.0,
                 dofmin = 0.1, dofmax = 5.0, qmax = None, alteqnull = None,abserr=None,fitdof=None,dof=None):
        '''
        Input:
        lrt             [Ntests] vector of test statistics
        a2 (optional)   [Ntests] vector of model variance parameters
        top (0.0)       tolerance for matching zero variance parameters or lrts
        scalemin (0.1)  minimum value used for fitting the scale parameter
        scalemax (5.0)  maximum value used for fitting scale parameter
        dofmin (0.1)    minimum value used for fitting the dof parameter
        dofmax (5.0)    maximum value used for fitting dof parameter
        qmax (None)      only the top qmax quantile is used for the fit
        '''
        self.lrt = lrt        
        self.alteqnull = alteqnull
        self.scale = None
        self.dof = dof
        self.mixture = None
        self.scalemin = scalemin
        self.scalemax = scalemax
        self.dofmin = dofmin
        self.dofmax = dofmax
        self.qmax = qmax
        self.tol = tol
        self.__fit_mixture()
        self.isortlrt = None
        self.abserr=abserr
        self.fitdof=fitdof
        #self.qlrt = self.lrt.argsort()

    def __fit_mixture(self):
        '''
        fit the mixture component
        '''
        if self.tol<0.0:
            logging.info('tol has to be larger or equal than zero.')
        if self.alteqnull is None:
            self.alteqnull=self.lrt==0
            logging.info("WARNING: alteqnull not provided, so using alteqnull=(lrt==0)")
        if self.mixture is None:
            self.mixture = 1.0-(sp.array(self.alteqnull).sum()*1.0)/(sp.array(self.alteqnull).shape[0]*1.0)
        return self.alteqnull, self.mixture
     

    def fit_params_Qreg(self):
        '''
        Fit the scale and dof parameters of the model by minimizing the squared error between
        the model log quantiles and the log P-values obtained on the lrt values.

        Only the top qmax quantile is being used for the fit (self.qmax is used in fit_scale_logP).
        '''
        #imin= sp.argsort(self.lrt[~self.i0])
        #ntests = self.lrt.shape[0]  
        if self.isortlrt is None:
            self.isortlrt = self.lrt.argsort()[::-1]            
            self.qnulllrtsort = (0.5+sp.arange(self.mixture*self.isortlrt.shape[0]))/(self.mixture*self.isortlrt.shape[0])   
            self.lrtsort = self.lrt[self.isortlrt]      
        resmin=[None] #CL says it had to be a list or wouldn't work, even though doesn't make sense
        if self.fitdof: #fit both scale and dof
            def f(x):
                res = self.fit_scale_logP(dof=x)
                if (resmin[0] is None) or (res['mse']<resmin[0]['mse']):
                    resmin[0]=res
                return res['mse']                   
        else:
            def f(x): #fit only scale                
                scale = x                        
                mse,imax=self.scale_dof_obj(scale,self.dof)
                if (resmin[0] is None) or (resmin[0]['mse']>mse):
                    resmin[0] = { #bookeeping for CL's mingrid.minimize1D
                        'mse':mse,
                        'dof':self.dof,
                        'scale':scale,
                        'imax':imax,
                    }                
                return mse 
        min = mingrid.minimize1D(f=f, nGrid=10, minval=self.dofmin, maxval=self.dofmax )
        self.dof = resmin[0]['dof']
        self.scale = resmin[0]['scale']
        self.imax=resmin[0]['imax']
        return resmin[0]       

    def fit_scale_logP(self, dof = None):        
        '''
        Extracts the top qmax lrt values to do the fit.        
        '''              
      
        if dof is None:
            dof =  self.dof
        resmin = [None]                       
       
        def f(x):            
            scale = x 
            err,imax=self.scale_dof_obj(scale,dof)
            if (resmin[0] is None) or (resmin[0]['mse']>err):
                resmin[0] = { #bookeeping for CL's mingrid.minimize1D
                    'mse':err,
                    'dof':dof,
                    'scale':scale,
                    'imax':imax,
                }
            return err
        min = mingrid.minimize1D(f=f, nGrid=10, minval=self.scalemin, maxval=self.scalemax )        
        return resmin[0]
        
    def scale_dof_obj(self,scale,dof):            
        base=sp.exp(1) #fitted params are invariant to this logarithm base (i.e.10, or e)
        
        nfalse=(len(self.alteqnull)-sp.sum(self.alteqnull))

        imax = int(sp.ceil(self.qmax*nfalse))  #of only non zer dof component
        p = st.chi2.sf(self.lrtsort[0:imax]/scale, dof)            
        logp = sp.logn(base,p)
        r = sp.logn(base,self.qnulllrtsort[0:imax])-logp
        if self.abserr:
            err=sp.absolute(r).sum()            
        else:#mean square error
            err = (r*r).mean()              
        return err,imax

    def mse_qreg(self, scale,dof,lrt,base):                        
        '''
        For debugging: returns mse for particular scale and dof, given pre-filtered lrt
        '''        
        p = st.chi2.sf(lrt/scale, dof)            
        logp = sp.logn(base,p)
        r = sp.logn(base,self.qnulllrtsort[0:len(lrt)])-logp
        mse = (r*r).mean()                
        return mse

    def sf(self, lrt = None, alteqnull=None):
        '''
        compute the survival function of the mixture of scaled chi-square_0 and scaled chi-square_dof
        ---------------------------------------------------------------------------
        Input:
        lrt (optional)     compute survival function for the lrt statistics
                           if None, compute survival function for original self.lrt
        ---------------------------------------------------------------------------
        Output:
        pv                 P-values
        ---------------------------------------------------------------------------
        '''        
        if lrt is None:
            lrt = self.lrt
            alteqnull=self.alteqnull
        else:
            assert alteqnull is not None, "alteqnull is none, but lrt is not (seems unexpected, JL)"
        
        pv = st.chi2.sf(lrt/self.scale,self.dof)*self.mixture                        
        pv[sp.array(alteqnull)]=1.0
        return pv

    #OBSOLETE: but was known to work. Can delete once we get past problems with python code solved.
    def computePVmixtureChi2(lrt,a2=None, tol= 0.0, mixture=0.5, scale = 1.0, dof = 1.0):
        '''
        OBSOLETE: but was known to work. Can delete once we get past problems with python code solved.

        computes P-values for a mixture of a scaled Chi^2_dof and Chi^2_0 distributions.
        The mixture weight is estimated from the fraction of models, where the parameter is at the boundary.
        The scale and degrees of freedom (dof) of the scaled Chi^2_dof are estimated by maximum likelihood, if the parameters provided are set to None.
        Note that accurate estimation of the mixture coefficient needs a sufficiently large number of tests to be performed.
        The P-values are computed as mixture*(1.0-CDF_Chi^2_1(lrt))
        --------------------------------------------------------------------------
        Input:
        lrt     : [S] 1D array of likelihood ratio tests (2*ln(likelihood ratio))
        a2      : [S] 1D array, if specified then a2 is used to determine the Chi^2_0
                  component, else lrt is used (optional).
        tol     : cutoff for members of the Chi^2_0 component is a2/lrt 0+tol.
        mixture : the scaled Chi^2_dof1 mixture component, if this parameter is set
                  to None, it will be estimated by the fraction of the tests that have
                  the weight a2 at the boundary (a2=0.0, lrt=0.0)
        scale   : the scale parameter of the scaled Chi^2_dof, if set to None the
                  parameter will be determined by maximum likelihood. (default 1.0)
        dof     : the degrees of freedom of the scaled Chi^2_dof, if set to None the
                  parameter will be determined by maximum likelihood. (default 1.0)
        --------------------------------------------------------------------------
        Output:
        pv        : [S] 1D-array of P-values computed as mixture*(1.0-CDF_Chi^2_dof(scale,lrt))
        mixture   : mixture weight of the scaled Chi^2_dof component
        scale     : scale of the scaled Chi^2_dof distribution
        dof       : degrees of freedom of the scaled Chi^2_dof distribution
        i0        : indicator for Chi^2_0 P-values
        --------------------------------------------------------------------------
        '''
        raise Exception("made changes to use alteqnull and did not modify this code as it looks obsolete")
        loc = None
        chi2mix = chi2mixture()
        chi2mix.lrt=lrt
        if mixture is None:
            i0, mixture = chi2mix.fit_mixture(a2=a2, tol=tol)
        else:
            chi2mix.mixture = mixture
            if a2 is None:
                i0 = (lrt<=(0.0+tol))
            else:
                i0 = (a2<=(0.0+tol))

        N=(~i0).sum()
        sumX = (lrt[~i0]).sum()
        logsumX = (sp.log(lrt[~i0])).sum()
        if (dof is None) and (scale is None):
        #f is the Gamma likelihood with the scale parameter maximized analytically as a funtion of 0.5 * the degrees of freedom
            f = lambda (k): -1.0*(-N*sp.special.gammaln(k)-k*N*(sp.log(sumX)-sp.log(k)-sp.log(N)) + (k-1.0)*logsumX-k*N)
            #f_ = lambda(x): 1-N*N/(2.0*x*sumX)
            res = minimize1D(f, evalgrid = None, nGrid=10, minval=0.1, maxval = 3.0)
            dof = 2.0*res[0]
        elif dof is None:
            f = lambda (k) : -1.0*(-N*sp.special.gammaln(k)-k*N*sp.log(2.0*scale)+(k-1.0)*logsumX-sumX/(2.0*scale))
            res = minimize1D(f, evalgrid = None, nGrid=10, minval=0.1, maxval = 3.0)
            dof = 2.0*res[0]
        if scale is None:
            #compute condition for ML
            if (1.0-(N*N*dof)/(4.0*sumX)>0):
                logging.warn('Warning: positive second derivative: No maximum likelihood solution can be found for the scale. returning scale=1.0 and dof=1.0')
                scale = 1.0
                dof = 1.0

            else:
                scale = sumX/(N*dof)
        pv = mixture*(st.chi2.sf(lrt/scale,dof)) # Can use the Chi^2 CDF/SF to evaluate the scaled Chi^2 by rescaling the input.
        pv[i0]=1.0
        return (pv,mixture,scale,dof,i0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("generate chi-2 distributed values")
    scale = 3
    dof = 2
    mixture = 0.5
    ntests = 10000
    lrt = sp.zeros((ntests))
    lrttest = sp.zeros((ntests))
    for i in xrange(dof):
        x = sp.randn(ntests)
        xtest = sp.randn(ntests)
        lrt += scale * (x*x)
        lrttest += scale * (xtest*xtest)

    lrt[sp.random.permutation(ntests)[0:sp.ceil(ntests*mixture)]] = 0.0
    lrttest[sp.random.permutation(ntests)[0:sp.ceil(ntests*mixture)]] = 0.0

    qmax = 0.2
    logging.info(("create the distribution object, with qmax = %.4f" % qmax))
    mix = chi2mixture( lrt = lrt, a2 = None , qmax = 0.2) #object constructor

    logging.info("fit the parameter of the object by log-Pvalue quantile regression")
    import time
    t0 = time.time()
    res = mix.fit_params_Qreg() # paramter fitting
    t1 = time.time()
    logging.info(("done after %.4f seconds." % (t1-t0)))
    logging.info("the true scale is %.4f, the fitted scale is %.4f." % (scale,mix.scale))
    logging.info("the true dof is %.4f, the fitted dof is %.4f." % (dof,mix.dof))
    logging.info("the true mixture is %.4f, the fitted mixture is %.4f." % (mixture,mix.mixture))

    logging.info("evaluating the survival function to get P-values of the training lrt (variable pv)")
    pv = mix.sf()
    logging.info("evaluating the survival function to get P-values of the testing lrt (variable pvtest)")
    pvtest = mix.sf(lrt=lrttest)
    pass

