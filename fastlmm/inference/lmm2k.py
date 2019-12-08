from __future__ import absolute_import
from __future__ import print_function
import scipy as SP
import numpy as NP
import scipy.linalg as LA
import scipy.optimize as opt
import scipy.stats as ST
import scipy.special as SS
from fastlmm.util.mingrid import *
from fastlmm.util.util import *
import time
import pdb

from bin2kernel import Bin2Kernel
from bin2kernel import makeBin2KernelAsEstimator
from bin2kernel import Bin2KernelLaplaceLinearN
from bin2kernel import getFastestBin2Kernel
from bin2kernel import Bin2KernelEPLinearN

def rotate(A, rotationMatrix,transposeRot=True):
    '''
    Apply an eigenvalue decomposition rotation matrix (Assummes that the rotation-matrix has orthogonal cols)
        
    takes care of low rank structure in A
        
    result = eig[1].T * A
    '''
    [Nr,k] = rotationMatrix.shape
    [N,M] = A.shape
    if Nr!=N:
        raise Exception("rotation and A are not aligned A.shape =  [%i,%i], rotation.shape = [%i,%i]" % (N,M,Nr,k))
    if k>Nr:
        raise Exception("rotation matrix has more columns than rows =  [%i,%i], rotation.shape = [%i,%i]" % (N,M,Nr,k))
        
    res = rotationMatrix.T.dot(A)

    if k<Nr:
        resLowRank = A - rotationMatrix.dot(res)
        res = [res,resLowRank]
    return res

def rotSymm(A,eig,delta=1.0,gamma=1.0, exponent = -0.5, forceSymm = True):
    '''
    do the rotation with K_0 to get a matrix square root
    '''
    [N,M]=A.shape
    [Nr,k] = eig[1].shape
    if Nr!=N:
        raise Exception("rotation and A are not aligned A.shape =  [%i,%i], rotation.shape = [%i,%i]" % (N,M,Nr,k))
    if symmetric and N!=M:
        raise Exception("A is not symmetric. A.shape =  [%i,%i], rotation.shape = [%i,%i]" % (N,M,Nr,k))
    if k>Nr:
        raise Exception("rotation matrix has more columns than rows =  [%i,%i], rotation.shape = [%i,%i]" % (N,M,Nr,k))

    res = eig[1].T.dot(A)
    deltaPow = myPow(delta,exponent,minVal)
    diag = eig[0]*gamma + delta
    diag = myPow(diag,exponent,minVal=0.0)
    if Nr>k:
        diag-=deltaPow
        res = dotDiag(res,diag)
        res = A-eig[1].dot(res)
    else:
        res = dotDiag(res,diag)
        if forceSymm:
            res = eig[1].dot(res)
    return res

def dotDiag(A, diag, minVal = None, symmetric = False, exponent = 1.0):
    '''
    Multiply by a diagonal matrix build from
        
    result = diag.dot(A)
    ''' 
    [N,M] = A.shape
    [Nd] = diag.shape
    if symmetric and N!=M:
        raise Exception("A is not symmetric. A.shape =  [%i,%i], rotation.shape = [%i,%i]" % (N,M,Nr,k))
    if Nd!=N:
        raise Exception("Matrices misaligned A.shape =  [%i,%i], diag.shape = [%i,%i]" % (N,M,Nr,k))
        
    diagcorr = myPow(diag,exponent,minVal)

    result = A * NP.lib.stride_tricks.as_strided(diagcorr, (diagcorr.size,A.shape[1]), (diagcorr.itemsize,0))
    if symmetric:
        result = dotDiag(A=A.T,diag=diagcorr,checkPos=False,minVal = minVal,symmetric = False, exponent = 1.0).T
    return result

def myPow(A,exponent,minVal=None):
    '''
    compute a power, with efficient computation of common powers
    '''
    if exponent == 1.0:
        Aexp = A
    elif exponent == -1.0:
        Aexp = 1.0/A
    elif exponent == 2.0:
        Aexp = A*A
    elif exponent == 0.5:
        Aexp = NP.sqrt(A)
    elif exponent == -0.5:
        Aexp = 1.0/NP.sqrt(A)
    elif exponent == -2.0:
        Aexp = 1.0/(A*A)
    else:
        Aexp = NP.power(A,exponent)
    if minVal is not None:
        i_Null = A<=minVal
        Aexp[i_Null]=0.0
    return Aexp

class lmm2k(object):
    '''
    linear mixed model with up to two kernels
    N(y | X*beta ; sigma2(gamma0*K0 + gamma1*K1 + I ) ),
    where
    K0 = G0*G0^T
    K1 = G1*G1^T
    '''
    __slots__ = ["G0","G1","Y","X","K0","K1","K","U","S","UX","Uy","UUX","UW","UUW","UUy","pos0","pos1","gamma0","gamma1","delta","exclude_idx","forcefullrank","numcalls","Xstar","Kstar","Kstar_star","UKstar","UUKstar","Gstar"]

    def __init__(self,forcefullrank=False):
        '''
        Input:
        forcefullrank   : if True, then the code always computes K and runs cubically
                            (False)
        '''
        self.X=None
        self.Y=None
        self.G0=None
        self.G1=None
        self.K0=None
        self.K1=None
        self.gamma0=1.0
        self.gamma1=1.0
        self.delta=1.0
        self.eig0=None
        self.eig1=None
        self.Yrot = None
        self.Xrot = None
        self.K1rot = None
        self.G1rot = None

    def setX(self, X):
        '''
        set the fixed effects X (covariates).
        The Kernel has to be set in advance by first calling setG() or setK().
        --------------------------------------------------------------------------
        Input:
        X       : [N*D] 2-dimensional array of covariates
        --------------------------------------------------------------------------
        '''
        self.X    = X
        self.Xrot = None
        
    def setY(self, Y):
        '''
        set the phenotype y.
        The Kernel has to be set in advance by first calling setG() or setK().
        --------------------------------------------------------------------------
        Input:
        Y       : [NxP] P-dimensional array of phenotype values
        --------------------------------------------------------------------------
        '''
        self.Y    = Y
        self.Yrot = None
        
    def setG0(self, G0):
        '''
        set the Kernel K0 from G0.
        This has to be done before setting the data setX() and setY(). 
        ----------------------------------------------------------------------------
        Input:
        G0              : [N*k0] array of random effects
        -----------------------------------------------------------------------------
        '''
        k = G0.shape[1]
        N = G0.shape[0]
        self.eig0 = None
        self.eig1 = None
        self.K1rot = None
        self.Yrot = None
        self.Xrot = None
        
        if ((not self.forcefullrank) and (k<N)):
            self.G0 = G0
        else:
            K0=G0.dot(G0.T);
            self.setK0(K0=K0)
        
    def setK0(self, K0):
        '''
        set the background Kernel K0.
        --------------------------------------------------------------------------
        Input:
        K0 : [N*N] array, random effects covariance (positive semi-definite)
        --------------------------------------------------------------------------
        '''
        self.K0    = K0
        self.G1    = None
        self.eig0  = None
        self.eig1  = None
        self.K1rot = None
        self.Yrot  = None
        self.Xrot  = None
        self.G1rot = None
        self.K1rot = None

    def setG1(self, G1):
        '''
        set the Kernel K1 from G1.
        ----------------------------------------------------------------------------
        Input:
        G0              : [N*k0] array of random effects
        -----------------------------------------------------------------------------
        '''
        k = self.G1.shape[1]
        N = self.G0.shape[0]
        self.eig1 = None
        self.G1rot = None
        if ((not self.forcefullrank) and (k<N)):
            #it is faster using the eigen decomposition of G.T*G but this is more accurate
            self.G1 = G1
        else:
            K1=self.G1.dot(self.G1.T);
            self.setK1(K1=K1)
        pass

    def setK1(self, K1):
        '''
        set the foreground Kernel K1.
        This has to be done before setting the data setX() and setY().
        --------------------------------------------------------------------------
        Input:
        K1 : [N*N] array, random effects covariance (positive semi-definite)
        --------------------------------------------------------------------------
        '''
        self.eig1 = None
        self.G1   = None
        self.K1   = K1
        self.K1rot= None
        self.G1rot= None
        
    def setVariances(self,gamma0=None,gamma1=None,delta=None):
        if gamma0 is not None:
            #background model changed, foreground rotation (U1,S1) changes:
            self.gamma0 = gamma0
            self.eig1 = None
            self.Xrot = None
            self.Yrot = None
            self.K1rot = None
            self.G1rot = None

        if delta is not None:
            #background model changed, foreground rotation (U1,S1) changes:
            self.delta = delta
            self.eig1 = None
            self.Xrot = None
            self.Yrot = None
            self.K1rot = None
            self.G1rot = None

        if gamma1 is not None:
            #foreground model changed
            self.gamma1 = gamma1




    def getEig1(self):
        '''
        '''
        if self.eig1 is None:
            #compute eig1
            if self.K1 is not None:
                if self.K1rot is None:
                    self.K1rot = rotSymm(self.K1, eig = self.eig0, exponent = -0.5, gamma=self.gamma0,delta = self.delta,forceSymm = False)
                    self.K1rot = rotSymm(self.K1rot.T, eig = self.eig0, exponent = -0.5, gamma=self.gamma0,delta = self.delta,forceSymm = False)
                self.eig1 = LA.eigh(self.K1rot)
            elif self.G1 is not None:
                [N,k] = self.G1.shape
                if self.G1rot is None:
                    self.G1rot = rotSymm(self.G1, eig = self.eig0, exponent = -0.5, gamma=self.gamma0,delta = self.delta,forceSymm = False)
                
                try:
                    [U,S,V] = LA.svd(self.G1rot,full_matrices = False)
                    self.eig1 = [S*S,U]
                except LA.LinAlgError:  # revert to Eigenvalue decomposition
                    print("Got SVD exception, trying eigenvalue decomposition of square of G. Note that this is a little bit less accurate")
                    [S_,V_] = LA.eigh(self.G1rot.T.dot(self.G1rot))
                    S_nonz=(S_>0.0)
                    S1 = S_[S_nonz]
                    U1=self.G1rot.dot(V_[:,S_nonz]/SP.sqrt(S1))
                    self.eig1=[S1,U1]
        return self.eig1

    def getEig0(self):
        '''
        '''
        if self.eig0 is None:
            #compute eig0
            if self.K0 is not None:
                self.eig1 = LA.eigh(self.K0)
            elif self.G1 is not None:
                [N,k] = self.G0.shape
                try:
                    [U,S,V] = LA.svd(self.G0,full_matrices = False)
                    self.eig0 = [S*S,U]
                except LA.LinAlgError:  # revert to Eigenvalue decomposition
                    print("Got SVD exception, trying eigenvalue decomposition of square of G. Note that this is a little bit less accurate")
                    [S_,V_] = LA.eigh(self.G0.T.dot(self.G0))
                    S_nonz=(S_>0.0)
                    S0 = S_[S_nonz]
                    U0=self.G0.dot(V_[:,S_nonz]/SP.sqrt(S0))
                    self.eig0=[S0,U0]
        return self.eig1

    def set_exclude_idx(self, idx):
        '''
        Set the indices of SNPs to be removed
        --------------------------------------------------------------------------
        Input:
        idx  : [k_up: number of SNPs to be removed] holds the indices of SNPs to be removed
        --------------------------------------------------------------------------
        '''        
        self.exclude_idx = idx
        
        
    def findGamma1givenGamma0(self, gamma0 = 0.0, nGridGamma1=10, minGamma1=0.0, maxGamma1=10000.0, **kwargs):
        '''
        Find the optimal h2 for a given K. Note that this is the single kernel case. So there is no a2.
        (default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance)
        --------------------------------------------------------------------------
        Input:
        nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at
        minH2   : minimum value for h2 optmization
        maxH2   : maximum value for h2 optmization
        --------------------------------------------------------------------------
        Output:
        dictionary containing the model parameters at the optimal h2
        --------------------------------------------------------------------------
        '''
        self.setVariances(gamma0 = gamma0)
        resmin=[None]
        def f(x,resmin=resmin,**kwargs):
            #self.setVariances(gamma1=x)
            res = self.nLLeval(gamma1=x,**kwargs)
            if (resmin[0] is None) or (res['nLL']<resmin[0]['nLL']):
                resmin[0]=res
                self.setGamma1[x]
            return res['nLL']
        min = minimize1D(f=f, nGrid=nGridGamma1, minval=minGamma1, maxval=maxGamma1 )
        return resmin[0]

    def findGammas(self, nGridGamma0=10, minGamma0=0.0, maxGamma0=10000.0, nGridGamma1=10, minGamma1=0.0, maxGamma1=10000.0,verbose=False, **kwargs):
        '''
        Find the optimal gamma0 and gamma1, such that K=gamma0*K0+gamma1*K1. Performs a double loop optimization (could be expensive for large grid-sizes)
        --------------------------------------------------------------------------
        Input:
        nGridGamma0 : number of a2-grid points to evaluate the negative log-likelihood at
        minGamma0   : minimum value for a2 optmization
        maxGamma0   : maximum value for a2 optmization
        nGridGamma1 : number of h2-grid points to evaluate the negative log-likelihood at
        minGamma1   : minimum value for h2 optmization
        maxGamma1   : maximum value for h2 optmization
        --------------------------------------------------------------------------
        Output:
        dictionary containing the model parameters at the optimal gamma0 and gamma1
        --------------------------------------------------------------------------
        '''
        self.numcalls=0
        resmin=[None]
        def f(x,resmin=resmin, nGridGamma1=nGridGamma1, minGamma1=minGamma1, maxGamma1=maxGamma1,**kwargs):
            self.numcalls+=1
            t0=time.time()
            res = self.findGamma1givenGamma0(gamma0=x, nGridGamma1=nGridGamma1, minGamma1=minGamma1, maxGamma1=maxGamma1,**kwargs)
            if (resmin[0] is None) or (res['nLL']<resmin[0]['nLL']):
                resmin[0]=res
            t1=time.time()
            #print "one objective function call took %.2f seconds elapsed" % (t1-t0)
            #import pdb; pdb.set_trace()
            return res['nLL']
        if verbose: print("findGammas")
        min = minimize1D(f=f, nGrid=nGridGamma1, minval=minGamma1, maxval=maxGamma1,verbose=False)
        #print "numcalls to innerLoopTwoKernel= " + str(self.numcalls)
        return resmin[0]


    def nLLeval(self,REML=True, gamma1 = None, dof = None, scale = 1.0):
        '''
        evaluate -ln( N( U^T*y | U^T*X*beta , h2*S + (1-h2)*I ) ),
        where ((1-a2)*K0 + a2*K1) = USU^T
        --------------------------------------------------------------------------
        Input:
        h2      : mixture weight between K and Identity (environmental noise)
        REML    : boolean
                  if True   : compute REML
                  if False  : compute ML
        dof     : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        logdelta: log(delta) allows to optionally parameterize in delta space
        delta   : delta     allows tomoptionally parameterize in delta space
        scale   : Scale parameter the multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        Output dictionary:
        'nLL'       : negative log-likelihood
        'sigma2'    : the model variance sigma^2
        'beta'      : [D*1] array of fixed effects weights beta
        'h2'        : mixture weight between Covariance and noise
        'REML'      : True: REML was computed, False: ML was computed
        'a2'        : mixture weight between K0 and K1
        'dof'       : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        '''
        if gamma1 is None:
            gamma1=self.gamma1
        if (gamma1<0.0):
            return {'nLL':3E20,
                    'gamma1':gamma1,
                    'delta' : self.delta,
                    'gamma0': self.gamma0,
                    'dof' : dof,
                    'REML':REML,
                    'scale':scale}
        k=self.S1.shape[0]
        N=self.Y.shape[0]
        D=self.X.shape[1]

        Sd = (gamma1*self.S1+1.0)*scale
        UXS = dotDiag(self.UX, Sd, minVal = 0.0, symmetric = False, exponent = -1.0)
        UYS = dotDiag(self.UY, Sd, minVal = 0.0, symmetric = False, exponent = -1.0)

        XKX = UXS.T.dot(self.UX)
        XKy = UXS.T.dot(self.Uy)
        yKy = UyS.T.dot(self.Uy)

        logdetK = SP.log(Sd).sum()


        if (k<N):#low rank part
        
            # determine normalization factor
            denom = (1.0*scale)
            

            XKX += self.UUX.T.dot(self.UUX)/(denom)
            XKy += self.UUX.T.dot(self.UUy)/(denom)
            yKy += self.UUy.T.dot(self.UUy)/(denom)      
            logdetK+=(N-k) * SP.log(denom)
 
        # proximal contamination (see Supplement Note 2: An Efficient Algorithm for Avoiding Proximal Contamination)
        # available at: http://www.nature.com/nmeth/journal/v9/n6/extref/nmeth.2037-S1.pdf
        # exclude SNPs from the RRM in the likelihood evaluation
        

        if len(self.exclude_idx) > 0:
            raise Exception("not implemented")          
            num_exclude = len(self.exclude_idx)
            
            # consider only excluded SNPs
            G_exclude = self.G[:,self.exclude_idx]
            
            self.UW = self.U.T.dot(G_exclude) # needed for proximal contamination
            UWS = self.UW / NP.lib.stride_tricks.as_strided(Sd, (Sd.size,num_exclude), (Sd.itemsize,0))
            assert UWS.shape == (k, num_exclude)
            
            WW = NP.eye(num_exclude) - UWS.T.dot(self.UW)
            WX = UWS.T.dot(self.UX)
            Wy = UWS.T.dot(self.Uy)
            assert WW.shape == (num_exclude, num_exclude)
            assert WX.shape == (num_exclude, D)
            assert Wy.shape == (num_exclude,)
            
            if (k<N):#low rank part
            
                self.UUW = G_exclude - self.U.dot(self.UW)
                
                WW += self.UUW.T.dot(self.UUW)/denom
                WX += self.UUW.T.dot(self.UUX)/denom
                Wy += self.UUW.T.dot(self.UUy)/denom
            
            
            #TODO: do cholesky, if fails do eigh
            # compute inverse efficiently
            [S_WW,U_WW] = LA.eigh(WW)
            
            UWX = U_WW.T.dot(WX)
            UWy = U_WW.T.dot(Wy)
            assert UWX.shape == (num_exclude, D)
            assert UWy.shape == (num_exclude,)
            
            # compute S_WW^{-1} * UWX
            WX = UWX / NP.lib.stride_tricks.as_strided(S_WW, (S_WW.size,UWX.shape[1]), (S_WW.itemsize,0))
            # compute S_WW^{-1} * UWy
            Wy = UWy / S_WW
            # determinant update
            logdetK += SP.log(S_WW).sum()
            assert WX.shape == (num_exclude, D)
            assert Wy.shape == (num_exclude,)
            
            # perform updates (instantiations for a and b in Equation (1.5) of Supplement)
            yKy += UWy.T.dot(Wy)
            XKy += UWX.T.dot(Wy)
            XKX += UWX.T.dot(WX)
            

        #######

        [SxKx,UxKx]= LA.eigh(XKX)
        i_pos = SxKx>1E-10
        beta = SP.dot(UxKx[:,i_pos],(SP.dot(UxKx[:,i_pos].T,XKy)/SxKx[i_pos]))

        r2 = yKy-XKy.dot(beta)

        if dof is None:#Use the Multivariate Gaussian
            if REML:
                XX = self.X.T.dot(self.X)
                [Sxx,Uxx]= LA.eigh(XX)
                logdetXX  = SP.log(Sxx).sum()
                logdetXKX = SP.log(SxKx).sum()
                sigma2 = r2 / (N - D)
                nLL =  0.5 * ( logdetK + logdetXKX - logdetXX + (N-D) * ( SP.log(2.0*SP.pi*sigma2) + 1 ) )
            else:
                sigma2 = r2 / (N)
                nLL =  0.5 * ( logdetK + N * ( SP.log(2.0*SP.pi*sigma2) + 1 ) )
            result = {
                  'nLL':nLL,
                  'sigma2':sigma2,
                  'beta':beta,
                  'gamma1':gamma1,
                  'REML':REML,
                  'gamma0':self.gamma0,
                  'delta':self.delta,
                  'scale':scale
                  }
        else:#Use multivariate student-t
            if REML:
                XX = self.X.T.dot(self.X)
                [Sxx,Uxx]= LA.eigh(XX)
                logdetXX  = SP.log(Sxx).sum()
                logdetXKX = SP.log(SxKx).sum()

                nLL =  0.5 * ( logdetK + logdetXKX - logdetXX + (dof + (N-D)) * SP.log(1.0+r2/dof) )
                nLL += 0.5 * (N-D)*SP.log( dof*SP.pi ) + SS.gammaln( 0.5*dof ) - SS.gammaln( 0.5* (dof + (N-D) ))
            else:
                nLL =   0.5 * ( logdetK + (dof + N) * SP.log(1.0+r2/dof) )
                nLL +=  0.5 * N*SP.log( dof*SP.pi ) + SS.gammaln( 0.5*dof ) - SS.gammaln( 0.5* (dof + N ))
            result = {
                  'nLL':nLL,
                  'dof':dof,
                  'sigma2':sigma2,
                  'beta':beta,
                  'gamma1':gamma1,
                  'REML':REML,
                  'gamma0':self.gamma0,
                  'delta':self.delta,
                  'scale':scale
                  }

        return result

