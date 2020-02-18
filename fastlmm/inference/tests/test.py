from __future__ import absolute_import
import numpy as NP
import scipy as SP
import scipy.io as SIO
import scipy.linalg as LA
from fastlmm.inference.laplace import LaplaceGLMM_N3K1, LaplaceGLMM_N1K3
from fastlmm.inference.ep import EPGLMM_N3K1, EPGLMM_N1K3
from fastlmm.inference import getLMM
import unittest
import os.path
import logging
from six.moves import range

currentFolder = os.path.dirname(os.path.realpath(__file__))


class TestLmmKernel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self._N = 50
        self._D = 1
        self._M = 2
        self._k0 = 5
        self._k1 = 10
        self._a2 = 0.4
        self._logdelta = +1
        self._beta = SP.array([1.2])

        from numpy.random import RandomState
        randomstate = RandomState(621360)
        self._X = randomstate.randn(self._N,self._D)
        self._G0 = randomstate.randn(self._N,self._k0)
        self._G1 = randomstate.randn(self._N,self._k1)
        self._y = randomstate.randn(self._N)

        self._Xstar = randomstate.randn(self._M,self._D)
        self._G0star = randomstate.randn(self._M,self._k0)
        self._G1star = randomstate.randn(self._M,self._k1)


    def test_predictions(self):
        model = getLMM()
        model.setG(G0=self._G0,G1=self._G1,a2=self._a2)
        model.setX(self._X)
        model.sety(self._y)

        # logdelta space
        model.setTestData(Xstar=self._Xstar,G0star=self._G0star,G1star=self._G1star)
        ystar = model.predictMean(self._beta,logdelta=self._logdelta)
        Gstar=SP.concatenate((SP.sqrt(1.0-self._a2) * self._G0star, SP.sqrt(self._a2) * self._G1star),1)
        weights = model.getPosteriorWeights(beta=self._beta,logdelta=self._logdelta)
        ystar2 = SP.dot(self._Xstar,self._beta) + SP.dot(Gstar,weights)
        self.assertAlmostEqual(ystar[0],ystar2[0])
        self.assertAlmostEqual(ystar[1],ystar2[1])

        # h2 space
        #TODO: this passes on the above toy data but fails on realistic data, fix!
        #h2 = 0.5
        #ystar = model.predictMean(self._beta,h2=h2)
        #weights = model.getPosteriorWeights(beta=self._beta,h2=h2)
        #ystar2 = SP.dot(self._Xstar,self._beta) + SP.dot(Gstar,weights)
        #self.assertAlmostEqual(ystar[0],ystar2[0])
        #self.assertAlmostEqual(ystar[1],ystar2[1])


    def test_nLLeval_1(self):
        """
        small regression test to check negative log likelihood function

        delta = 1, REML = False
        """

        model = getLMM()
        model.setG(G0=self._G0, G1=self._G1, a2=self._a2)
        model.setX(self._X)
        model.sety(self._y)
        result = model.nLLeval(REML=False,delta=1.0)

        target_result = {'scale': 1.0, 'h2': 0.0, 'beta': NP.array([ 0.05863443]), 'a2': 0.4, 'REML': False, 'nLL': 91.92983775736522, 'sigma2': 0.94826207355429604}

        # make sure results are the same
        for key in result.keys():
            NP.testing.assert_array_almost_equal(result[key], target_result[key])
            #self.assertAlmostEqual(result[key], target_result[key])


    def test_nLLeval_2(self):
        """
        small regression test to check negative log likelihood function

        delta = 1, REML = True
        """

        model = getLMM()
        model.setG(G0=self._G0, G1=self._G1 ,a2=self._a2)
        model.setX(self._X)
        model.sety(self._y)
        result = model.nLLeval(REML=True, delta=1.0)

        target_result = {'scale': 1.0, 'h2': 0.0, 'beta': NP.array([ 0.05863443]), 'a2': 0.4, 'REML': True, 'nLL': 90.940636012858121, 'sigma2': 0.96761436076968987}
        # make sure results are the same
        for key in result.keys():
            NP.testing.assert_array_almost_equal(result[key], target_result[key])
            #self.assertAlmostEqual(result[key], target_result[key])




    def test_nLLeval_3(self):
        """
        small regression test to check negative log likelihood function

        delta = None, h2 = 0.5, REML = True
        """

        model = getLMM()
        model.setG(G0=self._G0,G1=self._G1,a2=self._a2)
        model.setX(self._X)
        model.sety(self._y)
        result = model.nLLeval(REML=True, delta=None, h2=0.5)

        target_result = {'scale': 1.0, 'h2': 0.5, 'beta': NP.array([ 0.05863443]), 'a2': 0.4, 'REML': True, 'nLL': 90.940636012858121, 'sigma2': 1.9352287215393797}
        # make sure results are the same
        for key in result.keys():
            NP.testing.assert_array_almost_equal(result[key], target_result[key])
            #self.assertAlmostEqual(result[key], target_result[key])


class TestProximalContamination(unittest.TestCase):


    def test_one_kernel_fullrank(self):
        """
        test for one kernel only, using delta=1.0, REML=False
        """
        delta = 1.0

        # full rank as N <= s_c
        N = 10 # number of individuals
        d = 1 # number of fix effects
        s_c = 40 # number of SNPs used to construct the genetic similarity matrix

        X, y, G0, G1, G0_small, exclude_idx = generate_random_data(N, d, s_c)

        lmm_nocut = getLMM()
        lmm_nocut.setG(G0_small)
        lmm_nocut.setX(X)
        lmm_nocut.sety(y)
        ret_nocut = lmm_nocut.nLLeval(REML=False,delta=delta)

        lmm_nocut.setTestData(Xstar=X[:3],K0star=None,K1star=None,G0star=G0_small[:3],G1star=None)

        ypred_nocut = lmm_nocut.predictMean(beta=ret_nocut['beta'],delta=delta)

        lmm_cut = getLMM()
        lmm_cut.setG(G0)
        lmm_cut.setX(X)
        lmm_cut.sety(y)
        lmm_cut.set_exclude_idx(exclude_idx)
        ret_cut = lmm_cut.nLLeval(REML=False,delta=delta)

        lmm_cut.setTestData(Xstar=X[:3],G0star=G0[:3])
        ypred_cut = lmm_cut.predictMean(beta=ret_cut['beta'],delta=delta)

        # make sure results are the same
        for key in ret_nocut.keys():
            NP.testing.assert_array_almost_equal(ret_cut[key], ret_nocut[key])
            #self.assertAlmostEqual(ret_cut[key], ret_nocut[key])

        wproj = SP.random.randn(ypred_nocut.shape[0])
        self.assertAlmostEqual((wproj*ypred_nocut).sum(),(wproj*ypred_cut).sum())


    def test_one_kernel_lowrank(self):
        """
        test for one kernel only, using delta=1.0, REML=False
        """
        delta = 1.0

        # low rank as N > s_c
        N = 100 # number of individuals
        d = 1 # number of fix effects
        s_c = 40 # number of SNPs used to construct the genetic similarity matrix

        X, y, G0, G1, G0_small, exclude_idx = generate_random_data(N, d, s_c)

        lmm_nocut = getLMM()
        lmm_nocut.setG(G0_small)
        lmm_nocut.setX(X)
        lmm_nocut.sety(y)

        ret_nocut = lmm_nocut.nLLeval(REML=False,delta=delta)
        lmm_nocut.setTestData(Xstar=X[:3],G0star=G0_small[:3])
        ypred_nocut = lmm_nocut.predictMean(beta=ret_nocut['beta'],delta=delta)

        lmm_cut = getLMM()
        lmm_cut.setG(G0)
        lmm_cut.setX(X)
        lmm_cut.sety(y)
        lmm_cut.set_exclude_idx(exclude_idx)

        ret_cut = lmm_cut.nLLeval(REML=False,delta=delta)
        lmm_cut.setTestData(Xstar=X[:3],G0star=G0[:3])
        ypred_cut = lmm_cut.predictMean(beta=ret_cut['beta'],delta=delta)

        # make sure results are the same
        for key in ret_nocut.keys():
            NP.testing.assert_array_almost_equal(ret_cut[key], ret_nocut[key])
            #self.assertAlmostEqual(ret_cut[key], ret_nocut[key])

        wproj = SP.random.randn(ypred_nocut.shape[0])
        self.assertAlmostEqual((wproj*ypred_nocut).sum(),(wproj*ypred_cut).sum())

    def test_two_kernels_fullrank(self):
        """
        two kernels, using delta=1.0, REML=False
        """
        delta = 1.0

        # full rank as N <= s_c
        N = 10 # number of individuals
        d = 1 # number of fix effects
        s_c = 40 # number of SNPs used to construct the genetic similarity matrix

        X, y, G0, G1, G0_small, exclude_idx = generate_random_data(N, d, s_c)

        lmm_nocut = getLMM()
        lmm_nocut.setG(G0_small, G1)
        lmm_nocut.setX(X)
        lmm_nocut.sety(y)

        ret_nocut = lmm_nocut.nLLeval(REML=False,delta=delta)
        lmm_nocut.setTestData(Xstar=X[:3],G0star=G0_small[:3],G1star=G1[:3])
        ypred_nocut = lmm_nocut.predictMean(beta=ret_nocut['beta'],delta=delta)

        lmm_cut = getLMM()
        lmm_cut.setG(G0, G1)
        lmm_cut.setX(X)
        lmm_cut.sety(y)
        lmm_cut.set_exclude_idx(exclude_idx)

        ret_cut = lmm_cut.nLLeval(REML=False,delta=delta)
        lmm_cut.setTestData(Xstar=X[:3],G0star=G0[:3],G1star=G1[:3])
        ypred_cut = lmm_cut.predictMean(beta=ret_cut['beta'],delta=delta)

        # make sure results are the same
        for key in ret_nocut.keys():
            NP.testing.assert_array_almost_equal(ret_cut[key], ret_nocut[key])
            #self.assertAlmostEqual(ret_cut[key], ret_nocut[key])
        wproj = SP.random.randn(ypred_nocut.shape[0])
        self.assertAlmostEqual((wproj*ypred_nocut).sum(),(wproj*ypred_cut).sum())

    def test_two_kernels_lowrank(self):
        """
        two kernels, using delta=1.0, REML=False
        """
        delta = 1.0

        # low rank as N > s_c
        N = 100 # number of individuals
        d = 1 # number of fix effects
        s_c = 40 # number of SNPs used to construct the genetic similarity matrix

        X, y, G0, G1, G0_small, exclude_idx = generate_random_data(N, d, s_c)

        lmm_nocut = getLMM()
        lmm_nocut.setG(G0_small, G1)
        lmm_nocut.setX(X)
        lmm_nocut.sety(y)

        ret_nocut = lmm_nocut.nLLeval(REML=False,delta=delta)
        lmm_nocut.setTestData(Xstar=X[:3],G0star=G0_small[:3],G1star=G1[:3])
        ypred_nocut = lmm_nocut.predictMean(beta=ret_nocut['beta'],delta=delta)

        lmm_cut = getLMM()
        lmm_cut.setG(G0, G1)
        lmm_cut.setX(X)
        lmm_cut.sety(y)
        lmm_cut.set_exclude_idx(exclude_idx)

        ret_cut = lmm_cut.nLLeval(REML=False,delta=delta)
        lmm_cut.setTestData(Xstar=X[:3],G0star=G0[:3],G1star=G1[:3])
        ypred_cut = lmm_cut.predictMean(beta=ret_cut['beta'],delta=delta)

        # make sure results are the same
        for key in ret_nocut.keys():
            NP.testing.assert_array_almost_equal(ret_cut[key], ret_nocut[key])
            #self.assertAlmostEqual(ret_cut[key], ret_nocut[key])
        wproj = SP.random.randn(ypred_nocut.shape[0])
        self.assertAlmostEqual((wproj*ypred_nocut).sum(),(wproj*ypred_cut).sum())

    def test_two_kernels_fullrank_REML(self):
        """
        two kernels, using delta=1.0, REML=True
        """
        delta = 2.0

        # full rank as N <= s_c
        N = 10 # number of individuals
        d = 1 # number of fix effects
        s_c = 40 # number of SNPs used to construct the genetic similarity matrix

        X, y, G0, G1, G0_small, exclude_idx = generate_random_data(N, d, s_c)

        lmm_nocut = getLMM()
        lmm_nocut.setG(G0_small, G1)
        lmm_nocut.setX(X)
        lmm_nocut.sety(y)

        ret_nocut = lmm_nocut.nLLeval(REML=True,delta=delta)

        lmm_cut = getLMM()
        lmm_cut.setG(G0, G1)
        lmm_cut.setX(X)
        lmm_cut.sety(y)
        lmm_cut.set_exclude_idx(exclude_idx)

        ret_cut = lmm_cut.nLLeval(REML=True,delta=delta)

        # make sure results are the same
        for key in ret_nocut.keys():
            NP.testing.assert_array_almost_equal(ret_cut[key], ret_nocut[key])
            #self.assertAlmostEqual(ret_cut[key], ret_nocut[key])


    def test_two_kernels_lowrank_REML(self):
        """
        two kernels, using delta=1.0, REML=True
        """
        delta = 0.5

        # low rank as N > s_c
        N = 100 # number of individuals
        d = 1 # number of fix effects
        s_c = 40 # number of SNPs used to construct the genetic similarity matrix

        X, y, G0, G1, G0_small, exclude_idx = generate_random_data(N, d, s_c)

        lmm_nocut = getLMM()
        lmm_nocut.setG(G0_small, G1)
        lmm_nocut.setX(X)
        lmm_nocut.sety(y)

        ret_nocut = lmm_nocut.nLLeval(REML=True,delta=delta)

        lmm_cut = getLMM()
        lmm_cut.setG(G0, G1)
        lmm_cut.setX(X)
        lmm_cut.sety(y)
        lmm_cut.set_exclude_idx(exclude_idx)

        ret_cut = lmm_cut.nLLeval(REML=True,delta=delta)

        # make sure results are the same
        for key in ret_nocut.keys():
            NP.testing.assert_array_almost_equal(ret_cut[key], ret_nocut[key])
            #self.assertAlmostEqual(ret_cut[key], ret_nocut[key])

def generate_random_data(N, d, s_c):
    """
    small helper to generate a random data set
    """


    num_excludes = s_c // 2
    s = s_c # total number of SNPs to be tested

    X = NP.ones((N, d))
    y = NP.random.rand(N)

    G0 = NP.random.rand(N, s_c)
    G1 = NP.random.rand(N, s)

    # exclude randomly
    perm = SP.random.permutation(s_c)
    exclude_idx = perm[:num_excludes]
    include_idx = perm[num_excludes:]
    G0_small = G0[:,include_idx]


    return X, y, G0, G1, G0_small, exclude_idx


class TestBin2Kernel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self._N = 10
        self._D = 2

        from numpy.random import RandomState
        randomstate = RandomState(621360)

        self._X = randomstate.normal(0, 1, (self._N, self._D))
        self._G0 = randomstate.normal(0, 1, (self._N, 5))
        self._G1 = randomstate.normal(0, 1, (self._N, 6))
        y = randomstate.rand(self._N)
        y[y<=0.5] = -1.0
        y[y>0.5] = 1.0
        self._y = y

    def test_rmargll(self):
        model = LaplaceGLMM_N3K1('logistic')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])

        self.assertAlmostEqual(-13.821329282675068, model._regular_marginal_loglikelihood())

        model = EPGLMM_N3K1('erf')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])

        self.assertAlmostEqual(-16.599719862714529, model._regular_marginal_loglikelihood())

    def test_margll(self):
        model = LaplaceGLMM_N3K1('logistic', 'l2')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        self.assertAlmostEqual(-14.635329282675063, model.marginal_loglikelihood())

        model = EPGLMM_N3K1('erf', 'l2')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        self.assertAlmostEqual(-17.413719862714526, model.marginal_loglikelihood())

    def test_rmargll_after_optimization(self):
        model = LaplaceGLMM_N3K1('logistic')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        model.optimize()

        self.assertAlmostEqual(-6.75677866924, model._regular_marginal_loglikelihood())

        model = EPGLMM_N3K1('erf')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        model.optimize()

        self.assertAlmostEqual(-6.75424440734, model._regular_marginal_loglikelihood())

        model = LaplaceGLMM_N3K1('logistic', 'l2')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        model.optimize()

        self.assertAlmostEqual(-6.7584132443836662, model._regular_marginal_loglikelihood())

        model = EPGLMM_N3K1('erf', 'l2')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        model.optimize()

        self.assertAlmostEqual(-6.7545709287754718, model._regular_marginal_loglikelihood())

    def test_setK_rmargll_after_optimization(self):
        model = LaplaceGLMM_N3K1('logistic')
        model.setK(NP.dot(self._G0, self._G0.T), NP.dot(self._G1, self._G1.T))
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        model.optimize()

        self.assertAlmostEqual(-6.75677866924, model._regular_marginal_loglikelihood())

        model = EPGLMM_N3K1('erf')
        model.setK(NP.dot(self._G0, self._G0.T), NP.dot(self._G1, self._G1.T))
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        model.optimize()

        self.assertAlmostEqual(-6.75424440734, model._regular_marginal_loglikelihood())

    def test_rmargll_gradient(self):
        def func(x, model):
            model.sig02 = x[0]
            model.sig12 = x[1]
            model.sign2 = x[2]
            model.beta = x[3:]
            return model._regular_marginal_loglikelihood()

        def grad(x, model):
            model.sig02 = x[0]
            model.sig12 = x[1]
            model.sign2 = x[2]
            model.beta = x[3:]
            return model._rmll_gradient()

        model = LaplaceGLMM_N3K1('logistic')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)

        self.assertAlmostEqual( 0.0, SP.optimize.check_grad(func, grad, NP.array([1.0,1.2,1.5,-3.0,0.4]), model), places=5 )

        model = EPGLMM_N3K1('erf')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)

        self.assertAlmostEqual( 0.0, SP.optimize.check_grad(func, grad, NP.array([1.0,1.2,1.5,-3.0,0.4]), model), places=5 )

    def test_margll_gradient(self):
        def func(x, model):
            model.sig02 = x[0]
            model.sig12 = x[1]
            model.sign2 = x[2]
            model.beta = x[3:]
            return model.marginal_loglikelihood()

        def grad(x, model):
            model.sig02 = x[0]
            model.sig12 = x[1]
            model.sign2 = x[2]
            model.beta = x[3:]
            return model._mll_gradient()

        model = LaplaceGLMM_N3K1('logistic', 'l2')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        self.assertAlmostEqual( 0.0, SP.optimize.check_grad(func, grad, NP.array([1.0,1.2,1.5,-3.0,0.4]), model), places=5 )

        model = EPGLMM_N3K1('erf', 'l2')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        self.assertAlmostEqual( 0.0, SP.optimize.check_grad(func, grad, NP.array([1.0,1.2,1.5,-3.0,0.4]), model), places=5 )

    def test_prediction(self):
        model = LaplaceGLMM_N3K1('logistic')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])

        ps = NP.array([0.30481858,0.22520247,0.78060709,0.44734337,0.58824651,0.052388,\
                       0.60337951,0.22886631,0.54169641,0.71888192])

        p = model.predict(self._X, self._G0, self._G1)
        for i in range(ps.shape[0]):
            self.assertAlmostEqual(ps[i], p[i], places=4)

        model = EPGLMM_N3K1('erf')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])

        ps = NP.array([0.35487175,  0.20959901,  0.77285232,  0.48305102,  0.59542164,\
                       0.02770879,  0.6205743 ,  0.14239838,  0.45330375,  0.77123692])

        p = model.predict(self._X, self._G0, self._G1)
        for i in range(ps.shape[0]):
            self.assertAlmostEqual(ps[i], p[i])

    def test_rmargll_linearn(self):
        model = LaplaceGLMM_N1K3('logistic')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])

        self.assertAlmostEqual(-13.821329282675068, model._regular_marginal_loglikelihood())

        model = EPGLMM_N1K3('erf')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])

        self.assertAlmostEqual(-16.5997198627, model._regular_marginal_loglikelihood())

    def testr_rmargll_after_optimization_linearn(self):
        model = LaplaceGLMM_N1K3('logistic')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        model.optimize()

        self.assertAlmostEqual(-6.75677866924, model._regular_marginal_loglikelihood())

        model = EPGLMM_N1K3('erf')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])
        model.optimize()

        self.assertAlmostEqual(-6.75424440734, model._regular_marginal_loglikelihood())

    def test_rmargll_gradient_linearn(self):
        def func(x, model):
            model.sig02 = x[0]
            model.sig12 = x[1]
            model.sign2 = x[2]
            model.beta = x[3:]
            return model._regular_marginal_loglikelihood()

        def grad(x, model):
            model.sig02 = x[0]
            model.sig12 = x[1]
            model.sign2 = x[2]
            model.beta = x[3:]
            return model._rmll_gradient()

        model = LaplaceGLMM_N1K3('logistic')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)

        self.assertAlmostEqual( 0.0, SP.optimize.check_grad(func, grad, NP.array([1.0,1.2,1.5,-3.0,0.4]), model), places=5 )

        model = EPGLMM_N1K3('erf')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)

        self.assertAlmostEqual( 0.0, SP.optimize.check_grad(func, grad, NP.array([1.0,1.2,1.5,-3.0,0.4]), model), places=5 )

    def test_prediction_linearn(self):
        model = LaplaceGLMM_N1K3('logistic')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])

        ps = NP.array([0.30483881,0.2252297,0.78056257,0.44734442,0.58824122,0.05238813,\
                       0.60337305,0.22889891,0.54169378,0.71885939])

        p = model.predict(self._X, self._G0, self._G1)
        for i in range(ps.shape[0]):
            self.assertAlmostEqual(ps[i], p[i])

        model = EPGLMM_N1K3('erf')
        model.setG(self._G0, self._G1)
        model.setX(self._X)
        model.sety(self._y)
        model.sig02 = 1.5
        model.sig12 = 0.5
        model.sign2 = 0.8
        model.beta = NP.array([2.0, -1.0])

        ps = NP.array([0.35487175,  0.20959901,  0.77285232,  0.48305102,  0.59542164,\
                       0.02770879,  0.6205743 ,  0.14239838,  0.45330375,  0.77123692])

        p = model.predict(self._X, self._G0, self._G1)
        for i in range(ps.shape[0]):
            self.assertAlmostEqual(ps[i], p[i])

def getTestSuite():
    """
    set up composite test suite
    """

    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestBin2Kernel)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestProximalContamination)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestLmmKernel)

    return unittest.TestSuite([suite1, suite2, suite3])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
