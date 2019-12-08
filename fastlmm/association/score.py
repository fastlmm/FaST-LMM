from __future__ import absolute_import
import numpy as NP
import scipy as sp
import scipy.linalg as LA
import numpy.linalg as nla
import os
import sys
import glob
from six.moves import range
sys.path.append("./../../pyplink")
from fastlmm.pyplink.plink import *
from pysnptools.util.pheno import *
from fastlmm.util.mingrid import *
#import pdb
import scipy.stats as ST
import fastlmm.util.stats as ss
import fastlmm.util.util as util
import fastlmm.association as association


class scoretest(association.varcomp_test):
    '''
    This is the super class that just performs a score test for the 1K linear case, gives P-values etc.
    All other models are inherited
    '''
    __slots__ = ["squaredform","expectationsqform","varsqform","GPG","GPY"]
    def __init__(self,Y,X=None,appendbias=False):
        association.varcomp_test.__init__(self,Y=Y,X=X,appendbias=appendbias)
        pass

    def _score(self,G1):
        '''
        This calls the score computation for a single kernel
        Christoph guess: varsqform is the variance of sigma_g, and is the inverse of the Fisher information wrt sigma_g
        Christoph: should compute variance of h2 (and test h2>0), which eliminates one nusiance parameter and yields a better test
        '''
        self.squaredform, self.expectationsqform, self.varsqform, self.GPG, self.GPY= scoreNoK( Y=self.Y, X = self.X, Xdagger=None, G = G1, sigma2=None,Bartletcorrection=True)
        if self.GPG.shape[0]==0:
            raise Exception("GPG is empty")
        return self.squaredform, self.expectationsqform, self.varsqform, self.GPG
        
    #def _pv(self, type): # this used to default to ="davies"
    #    evalstring = 'self.pv_%s(self.squaredform,self.expectationsqform,self.varsqform,self.GPG)' % (type)
    #    return  eval(evalstring)

    def testG(self,G1,type, altModel=None,i_exclude=None,G_exclude=None):
        """
        Params:
            G1:         SNPs to be tested
            type:       moment matching davies etc
            i_exclude:  Dummy
            G_exclude:  Dummy
        """    # this used to default to ="davies"        
        self._score(G1=G1)
        pv = type.pv(self.squaredform,self.expectationsqform,self.varsqform,self.GPG)
        #stat = scoretest.scoreteststat(self.squaredform,self.varsqform)
        test={
              'pv':pv,
              'stat':self.squaredform
              }
        return test
      


class scoretest_logit(scoretest):
    __slots__ = ["Y","X","Xdagger","logreg_result","logreg_mod","pY","stdY","VX","pinvVX"]
    
    def __init__(self,Y,X=None,appendbias=False):

        ## check if is binary        
        uniquey=sp.unique(Y)
        if not sp.sort(uniquey).tolist()==[0,1]: raise Exception("must use binary data in {0,1} for logit tests, found:" + str(Y))

        scoretest.__init__(self,Y=Y,X=X,appendbias=appendbias)
        #from sklearn.linear_model import LogisticRegression as LR
        #logreg_sk = LR(C=200000.0)
        #logreg_sk.fit( X, Y )
        import statsmodels.api as sm
        self.logreg_mod = sm.Logit(Y[:,0],X)
        self.logreg_result = self.logreg_mod.fit(disp=0)
        self.pY = self.logreg_result.predict(X)
        self.stdY=sp.sqrt(self.pY*(1.0-self.pY))
        self.VX=self.X * NP.lib.stride_tricks.as_strided((self.stdY), (self.stdY.size,self.X.shape[1]), (self.stdY.itemsize,0))
        self.pinvVX=nla.pinv(self.VX)

    def _score(self, G1):
        '''
        compute the score

        Inputs:
            Bartletcorrection: refers to dividing by N-D instead of D, it is used in REML
        Outputs:
            squaredform
            expectationsqform
            varsqform
            GPG=P^1/2*K*P^1/2 (take eigenvalues of this for Davies method)
        '''
        Y=self.Y  
        X=self.X  
        N=Y.shape[0]
        if Y.ndim == 1:
            P=1                                              #num of phenotypes
        else:
            P = Y.shape[1]
        if X is None:
            D = 1                                            #num of covariates (and assumes they are independent)
        else:
            D = X.shape[1]    
        RxY = (self.Y.flatten()-self.pY)       #residual of y regressed on X, which here, is equivalent to sigma2*Py (P is the projection matrix, which is idempotent)
        VG = G1 * NP.lib.stride_tricks.as_strided(self.stdY, (self.stdY.size,G1.shape[1]), (self.stdY.itemsize,0))
        GY = G1.T.dot(RxY)
        squaredform=(GY*GY).sum()/(2.0*P)
        
        RxVG,Xd =  linreg(VG, X=self.VX, Xdagger=self.pinvVX,rcond=-1)
        if (G1.shape[0]<G1.shape[1]):
            GPG=RxVG.dot(RxVG.T)/(2.0*P)
        else:
            GPG=RxVG.T.dot(RxVG)/(2.0*P)
        self.squaredform=squaredform
        self.expectationsqform=None
        self.varsqform=None
        self.GPG=GPG
        return squaredform, GPG

class scoretest2K(scoretest):
    __slots__ = ["K","PxKPx","G0","U","S","Xdagger","UY","UUY","YUUY","optparams","expectedinfo","lowrank","Neff"]
     
    def __init__(self,Y,X=None,K=None,G0=None,appendbias=False,forcefullrank=False):
        scoretest.__init__(self,Y=Y,X=X,appendbias=appendbias)
        self.Xdagger = None
        self.G0=G0
        self.K=K
        #compute the spectral decomposition of K
        self.lowrank = False
        N=Y.shape[0]
        if Y.ndim==1:
            P=1
        else:
            P=Y.shape[1]
        D=1
        if X is not None:
            D=X.shape[1]
        self.Neff = N-D
        if self.K is not None:            
            ar = sp.arange(self.K.shape[0])
            self.K[ar,ar]+=1.0
            self.PxKPx,self.Xdagger = linreg(Y=(self.K), X=self.X, Xdagger=self.Xdagger)
            self.PxKPx,self.Xdagger = linreg(Y=self.PxKPx.T, X=self.X, Xdagger=self.Xdagger)
            [self.S,self.U] = LA.eigh(self.PxKPx)
            self.K[ar,ar]-=1.0
            self.U=self.U[:,D:N]
            self.S=self.S[D:N]-1.0
        elif 0.7*(self.Neff)<=self.G0.shape[1] or forcefullrank:
            self.K = self.G0.dot(self.G0.T)
            # BR: changed K to self.K (K is not defined)
            ar = sp.arange(self.K.shape[0])
            self.K[ar,ar]+=1.0
            self.PxKPx,self.Xdagger = linreg(Y=(self.K), X=self.X, Xdagger=self.Xdagger)
            self.PxKPx,self.Xdagger = linreg(Y=self.PxKPx.T, X=self.X, Xdagger=self.Xdagger)
            self.K[ar,ar]-=1.0
            # BR: changed PxKPx to self.PxKPx (PxKPx is not defined)
            [self.S,self.U] = LA.eigh(self.PxKPx)
            self.U=self.U[:,D:N]
            self.S=self.S[D:N]-1.0
        else:
            PxG,self.Xdagger = linreg(Y=self.G0, X=self.X, Xdagger=self.Xdagger)
            [self.U,self.S,V] = LA.svd(PxG,False,True) #!!!use big_svd?
            inonzero = self.S>1E-10
            self.S=self.S[inonzero]*self.S[inonzero]
            self.U=self.U[:,inonzero]
            self.lowrank = True
            pass

        #rotate the phenotype as well as the fixed effects        
        self.UY = self.U.T.dot(self.Y)
        
        if self.lowrank:
            Yres,self.Xdagger = linreg(Y=self.Y, X=self.X, Xdagger=self.Xdagger)
            self.UUY = Yres-self.U.dot(self.UY)
            self.YUUY = (self.UUY * self.UUY).sum()
            pass
        
        #learn null model
        resmin=[None]
        def f(x,resmin=resmin,**kwargs):
            res = self._nLLeval(h2=x)
            if (resmin[0] is None) or (res['nLL']<resmin[0]['nLL']):
                resmin[0]=res
            return res['nLL']
        min = minimize1D(f, evalgrid = None, nGrid=20, minval=0.0, maxval = 0.99999)
        
        self.optparams = resmin[0]

        #pre-compute model parameters
        self.expectedinfo = sp.zeros((2,2))
        #tr(PIPI)
        Sd = 1.0/((1.0 - self.optparams['h2']) + self.optparams['h2'] * self.S) 
        Sd *= Sd
        self.expectedinfo[0,0] = (Sd).sum()#/(self.optparams['sigma2']*self.optparams['sigma2'])
        if self.lowrank:
            self.expectedinfo[0,0]+=((self.Neff-self.S.shape[0]))/((1.0 - self.optparams['h2'])*(1.0 - self.optparams['h2']))
        #tr(PKPI)
        Sd*=self.S
        self.expectedinfo[1,0] = (Sd).sum()#/(self.optparams['sigma2']*self.optparams['sigma2'])
        self.expectedinfo[0,1] = self.expectedinfo[1,0]
        #tr(PKPK)
        Sd*=self.S
        self.expectedinfo[1,1] = (Sd).sum()#/(self.optparams['sigma2']*self.optparams['sigma2'])
        self.expectedinfo*=0.5*P/(self.optparams['sigma2']*self.optparams['sigma2'])
        
        pass

    def _nLLeval(self,h2=0.0):
        '''
        evaluate -ln( N( U^T*y | U^T*X*beta , h2*S + (1-h2)*I ) ),
        where K = USU^T
        --------------------------------------------------------------------------
        Input:
        h2      : mixture weight between K and Identity (environmental noise)
        --------------------------------------------------------------------------
        Output dictionary:
        'nLL'       : negative log-likelihood
        'sigma2'    : the model variance sigma^2
        'h2'        : mixture weight between Covariance and noise
        --------------------------------------------------------------------------
        '''
        if (h2<0.0) or (h2>=1.0):
            return {'nLL':3E20,
                    'h2':h2
                    }
        k=self.S.shape[0]
        N=self.Y.shape[0]
        if self.Y.ndim==1:
            P=1
        else:
            P=self.Y.shape[1]
        
        Sd = h2*self.S + (1.0-h2)
        UYS = self.UY / NP.lib.stride_tricks.as_strided(Sd, (Sd.size,self.UY.shape[1]), (Sd.itemsize,0))
        
        YKY = (UYS*self.UY).sum()

        logdetK   = sp.log(Sd).sum()

        if (self.lowrank):#low rank part
            YKY += self.YUUY/(1.0-h2)
            logdetK +=sp.log(1.0-h2)*(self.Neff*P-k)
        
        sigma2 = YKY / (self.Neff*P)
        nLL =  0.5 * ( logdetK + self.Neff*P * ( sp.log(2.0*sp.pi*sigma2) + 1 ) )
        result = {
                  'nLL':nLL,
                  'sigma2':sigma2,
                  'h2':h2
                  }
        return result

    def _score(self, G1):
        '''
        compute the score with a background kernel
        '''
        #if 1:
        #    #background kernel
        #    self.K=self.G.dot(self.G.T)
        #    h2 = self.optparams['h2']
        #    sig = self.optparams['sigma2']
        #    V = h2*self.K + (1-h2)*sp.eye(self.K.shape[0])
        #    V*=sig
        #    Vi=LA.inv(V)
        #    P =LA.inv(self.X.T.dot(Vi).dot(self.X))
        #    P=self.X.dot(P.dot(self.X.T))
        #    P=Vi.dot(P.dot(Vi))
        #    Px = Vi-P                                    

        P = self.UY.shape[1]
        resG, Xdagger = linreg(Y=G1, X=self.X, Xdagger=self.Xdagger)
        sigma2e = (1.0-self.optparams["h2"])*self.optparams["sigma2"]
        sigma2g = self.optparams["h2"]*self.optparams["sigma2"]
        UG = self.U.T.dot(resG)
        if self.lowrank:
            UUG = resG-self.U.dot(UG)
        Sd = 1.0/(self.S*sigma2g + sigma2e)
        SUG = UG * NP.lib.stride_tricks.as_strided(Sd, (Sd.size,UG.shape[1]), (Sd.itemsize,0))
        #tr(YPGGPY)
        GPY = SUG.T.dot(self.UY)
        if self.lowrank:
            GPY += UUG.T.dot(self.UUY)/sigma2e
        squaredform = 0.5*(GPY*GPY).sum()
        #tr(PGG)
        if G1.shape[0]>G1.shape[1]:
            GPG = SUG.T.dot(UG)
        else:
            GPG = SUG.dot(UG.T)
        expectationsqform = 0.5*P*GPG.trace()
        #tr(PGGPGG)
        trPGGPGG = 0.5*P*(GPG*GPG).sum()
        #tr(PGGPI)
        SUG*=SUG
        expectedInfoCross=sp.empty(2)
        expectedInfoCross[0] = 0.5*P*SUG.sum()
        #tr(PGGPK)
        SUG*=NP.lib.stride_tricks.as_strided(self.S, (self.S.size,SUG.shape[1]), (self.S.itemsize,0))
        expectedInfoCross[1] = 0.5*P*SUG.sum()
        if self.lowrank:
            if G1.shape[0]>G1.shape[1]:
                GPG_lowr = UUG.T.dot(UUG)/sigma2e
            else:
                GPG_lowr = UUG.dot(UUG.T)/sigma2e
            GPG+=GPG_lowr
            #tr(PGGPGG)
            expectationsqform += 0.5*P*GPG_lowr.trace()
            trPGGPGG += 0.5*P*(GPG_lowr*GPG_lowr).sum()
            #tr(PGGPI)
            expectedInfoCross[0] += 0.5*P*GPG_lowr.trace()/(sigma2e) 
        varsqform = 1.0/(trPGGPGG - expectedInfoCross.dot(LA.inv(self.expectedinfo).dot(expectedInfoCross))) 
        self.squaredform = squaredform
        self.expectationsqform=expectationsqform
        self.varsqform=varsqform
        self.GPG = GPG*0.5       
        return self.squaredform, self.expectationsqform, self.varsqform, self.GPG

    def _findH2(self, nGridH2=10, minH2 = 0.0, maxH2 = 0.99999, **kwargs):
        '''
        Find the optimal h2 for a given K.
        (default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance) 
        --------------------------------------------------------------------------
        Input:
        nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at
        minH2   : minimum value for h2 optimization
        maxH2   : maximum value for h2 optimization
        --------------------------------------------------------------------------
        Output:
        dictionary containing the model parameters at the optimal h2
        --------------------------------------------------------------------------
        '''
        #f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
        resmin=[None]
        def f(x,resmin=resmin,**kwargs):
            res = self._nLLeval(h2=x,**kwargs)
            if (resmin[0] is None) or (res['nLL']<resmin[0]['nLL']):
                resmin[0]=res
            return res['nLL']
        min = minimize1D(f=f, nGrid=nGridH2, minval=minH2, maxval=maxH2 )
        return resmin[0]


def linreg(Y, X=None, Xdagger=None,rcond=-1):       
    if Y.ndim == 1:
        P=1
    else:
        P = Y.shape[1]    
    if X is None:
        RxY = Y-Y.mean(0)
        return RxY, None
    else:        
        if Xdagger is None:
            #Xdagger = LA.pinv(X,rcond) #can be ridiculously slow (solves a linear system), 20 seconds instead of 0.1 sec.
            Xdagger = nla.pinv(X)       #SVD-based, and seems fast
        RxY = Y-X.dot(Xdagger.dot(Y))
        return RxY, Xdagger
        
def scoreNoK( Y, X = None, Xdagger=None, G = None, sigma2=None,Bartletcorrection=True):
    '''
    compute the score

    Inputs:
        Bartletcorrection: refers to dividing by N-D instead of D, it is used in REML
    Outputs:
        squaredform
        expectationsqform
        varsqform
        GPG=P^1/2*K*P^1/2 (take eigenvalues of this for Davies method)
    '''    
    N=Y.shape[0]
    if Y.ndim == 1:
        P=1                                              #num of phenotypes
    else:
        P = Y.shape[1]
    if X is None:
        D = 1                                            #num of covariates (and assumes they are independent)
    else:
        D = X.shape[1]    
    RxY, Xdagger = linreg(Y=Y,X=X,Xdagger=Xdagger)       #residual of y regressed on X, which here, is equivalent to sigma2*Py (P is the projection matrix, which is idempotent)
    
    if sigma2 is None:                                   #     note: Xdagger is pseudo inverse of X, or (X^T*X)^1*X^T such that Xdagger*y=beta                                                   
        if Bartletcorrection:                            
            sigma2 = (RxY*RxY).sum()/((N-D)*P)
        else:
            sigma2 = (RxY*RxY).sum()/(N*P)
    
    RxG, Xdagger = linreg(Y=G,X=X, Xdagger = Xdagger)   #residual of G regressed on X, which here, is equivalent to PG (P is the projection matrix, and in this one kernel case, is idempotent)
                                                        #     note: P is never computed explicitly, only via residuals such as Py=1/sigma2(I-Xdagger*X)y and PG=1/sigma2(I-Xdagger*X)G
                                                        #     also note that "RxY"=Py=1/sigma2*(I-Xdagger*X)y is nothing more (except for 1/sigma2) than the residual of y regressed on X (i.e. y-X*beta), 
                                                        #     and similarly for PG="RxG"
    GtRxY = G.T.dot(RxY)
    squaredform = ((GtRxY*GtRxY).sum())*(0.5/(sigma2*sigma2))       # yPKPy=yPG^T*GPy=(yPG^T)*(yPG^T)^T
    if G.shape[0]>G.shape[1]:
        GPG = sp.dot(RxG.T,RxG)                                      #GPG is always a square matrix in the smaller dimension
    else:
        GPG = sp.dot(RxG,RxG.T)
    expectationsqform = P*(GPG.trace())*(0.5/sigma2)                 #note this is Trace(PKP)=Trace(PPK)=Trace(PK), for P=projection matrix in comment, and in the code P=1=#phen
    expectedinfo00 = P*(GPG*GPG).sum()*(0.5/(sigma2*sigma2))
    expectedinfo10 = expectationsqform/sigma2                       # P*0.5/(sigma2*sigma2)*GPG.trace()
    expectedinfo11 = P*(N-D)*(0.5/(sigma2*sigma2))
    varsqform = 1.0/(expectedinfo00 - expectedinfo10*expectedinfo10/expectedinfo11)
    #if 1:
    #    XXi=LA.inv(X.T.dot(X))
    #    Px=(sp.eye(N)-X.dot(XXi).dot(X.T))/sigma2
    #pdb.set_trace()
    GPG/=sigma2*2.0      #what we will take eigenvalues of for Davies (which is P^1/2*K*P^1/2)
    
    #for debugging, explicitly compute GPG=P^1/2 * K * P^1/2        
    #SigInv=(1/sigma2)*sp.eye(N,N)
    #Phat=X.dot(LA.inv(X.T.dot(SigInv).dot(X))).dot(X.T).dot(SigInv)
    #PP=SigInv.dot(sp.eye(N,N)-Phat)
    #K=G.dot(G.T)
    #PKP=PP.dot(K).dot(PP)
    #ss.stats(PKP-PKP.T)
    ##eigvalsFull=LA.eigh(PKP,eigvals_only=True)    
    #eigvalsFull2=LA.eigvals(PKP)    
    #eigvalsLow =LA.eigh(GPG,eigvals_only=True)    
    #GPG=PKP*0.5
    #pdb.set_trace()
    
    return squaredform, expectationsqform, varsqform, GPG, GtRxY*(0.25/sigma2)








if __name__ == "__main__":
    if 1:#example p-value computation for sample data
        
        #specify the directory that contains the data
        datadir = "data"#os.path.join('twokerneltest','data')
        #specify the directory that contains the alternative models in form of ped files
        datadiralt = os.path.join(datadir,'altmodels')
        pedfilesalt = glob.glob(os.path.join(datadiralt, '*.ped'))
        for i in range(len(pedfilesalt)):
            pedfilesalt[i]=pedfilesalt[i][0:-4]
        
        phenofile = os.path.join(datadir,'phen.N1000.M5000.txt')
        covarfile = os.path.join(datadir,'covariates.N1000.M5000.txt')
        #base0 = os.path.join(datadir,'snpDat.N1000.M5000.20Snps')
        base0 = os.path.join(datadir,'snpDat.N1000.M5000.10_20Snps')
        
        #specify index of the phenotype to be tested
        ipheno = 0  #only one phenotype in file, use the first one
        
        #exclusion parameters (correction for proximal contamination)
        mindist = 10  #minimum distance to alternative SNPs to be included in null model   
        idist = 2       #use genetic distance
        #idist = 3      #use basepair distance

        #run the example
        logging.info(('\n\n\nrunning real data example')        )
        logging.info(('base file of null model: %s' % base0))
        logging.info(('testing all SNP sets in %s' % datadiralt))
        result = testPedfilesFromDir(phenofile, base0, pedfilesalt, ipheno=ipheno, mindist = mindist, idist=idist, covarfile = covarfile)
        
