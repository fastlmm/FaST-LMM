from __future__ import print_function #Python 2 & 3 compatibility
from __future__ import absolute_import
import numpy as np
import logging
import unittest
import os
import scipy.linalg as LA
import time

from pysnptools.snpreader import Bed,Pheno
from pysnptools.snpreader import SnpData,SnpReader
from pysnptools.kernelreader import KernelNpz
from pysnptools.kernelreader import SnpKernel
from pysnptools.kernelreader import KernelReader
from pysnptools.kernelreader import Identity as KernelIdentity
import pysnptools.util as pstutil
from pysnptools.standardizer import DiagKtoN,UnitTrained
from pysnptools.standardizer import Unit
from pysnptools.util import intersect_apply
from pysnptools.standardizer import Standardizer
from fastlmm.inference.lmm import LMM
from pysnptools.standardizer import Identity as StandardizerIdentity
from scipy.stats import multivariate_normal
from fastlmm.util.pickle_io import load, save
from pysnptools.pstreader import PstReader
from six.moves import range

class _SnpWholeTest(KernelReader):
    '''
    Warning: Assumes that if train and test contains the same iid, they have the same value.
    '''
    def __init__(self,train,test,standardizer,block_size,iid0=None):
        self.train = train
        self.test = test
        self.standardizer = standardizer
        assert standardizer.is_constant, "Expect standardizer to be constant"
        self.block_size = block_size
        if iid0 is not None:
            _row = iid0

    @property
    def row(self):
        if not hasattr(self,'_row'):
            assert np.array_equal(self.train.sid,self.test.sid), "Expect train and test to have same sid in same order"
            train_set = set(tuple(item) for item in self.train.iid)
            test_unique = [item2 for item2 in (tuple(item) for item in self.test.iid) if item2 not in train_set]
            self._row = np.r_[self.train.iid,np.array(test_unique,dtype='str').reshape(-1,2)]
        return self._row


    @property
    def col(self):
        return self.test.iid

    def __getitem__(self, iid_indexer_and_snp_indexer):
        if isinstance(iid_indexer_and_snp_indexer,tuple):
            iid0_indexer, iid1_indexer = iid_indexer_and_snp_indexer
        else:
            iid0_indexer = iid_indexer_and_snp_indexer
            iid1_indexer = iid0_indexer

        row_index_or_none = PstReader._make_sparray_from_sparray_or_slice(self.row_count, iid0_indexer)
        col_index_or_none = PstReader._make_sparray_from_sparray_or_slice(self.col_count, iid1_indexer)

        if row_index_or_none is None:
            row_index_or_none = list(range(self.row_count))

        assert not isinstance(row_index_or_none,str), "row_index_or_none should not be a string"
        iid = self.row[row_index_or_none]

        if col_index_or_none is None or np.array_equal(col_index_or_none,list(range(self.col_count))):
            test = self.test
        else:
            test = self.test[col_index_or_none]
        
        try: #case 1: asking for train x test
            train = self.train[self.train.iid_to_index(iid),:]
            is_ok = True
        except:
            is_ok = False
        if is_ok:
            return _SnpTrainTest(train=train,test=test,standardizer=self.standardizer,block_size=self.block_size)

        #case 2: asking for train x test
        if np.array_equal(test.iid,iid):
            return SnpKernel(test,standardizer=self.standardizer,block_size=self.block_size)

        #case 3: Just re-reordering the iids
        if len(row_index_or_none) == self.row_count and (col_index_or_none is None or len(col_index_or_none) == self.col_count):
            result = _SnpWholeTest(train=self.train,test=test,standardizer=self.standardizer,block_size=self.block_size,iid0=iid)
            return result

        
        raise Exception("When reading from a _SnpWholeTest, can only ask to reorder iids or to access from train x test or test x test")


    #!!! does it make sense to read from disk in to parts?
    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        result = self[row_index_or_none,col_index_or_none]._read(row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok)
        return result

    def __repr__(self):
        s = "_SnpWholeTest(train={0},test={1},standardizer={2}".format(self.train,self.test,self.standardizer)
        if self.block_size is not None:
            s += ",block_size={0}".format(self.block_size)
        s += ")"
        return s

    def copyinputs(self, copier):
        #Doesn't need run_once
        copier.input(self.train)
        copier.input(self.test)
        copier.input(self.standardizer)

class _SnpTrainTest(KernelReader):
    def __init__(self,train,test,standardizer,block_size):
        self.train = train
        self.test = test
        self.standardizer = standardizer
        assert standardizer.is_constant, "Expect standardizer to be constant"
        self.block_size = block_size
        if np.array_equal(train.iid,test.iid):
            self._col = train.iid
        else:
            self._col = test.iid

    @property
    def row(self):
        return self.train.iid

    @property
    def col(self):
        return self._col

    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        assert self.train.sid_count == self.test.sid_count, "real assert"
        #case 1: asking for all of train x test
        if (row_index_or_none is None or np.array_equal(row_index_or_none,np.arange(self.row_count))
            and col_index_or_none is None or np.array_equal(col_index_or_none,np.arange(self.col_count))):

            #Do all-at-once (not in blocks) if 1. No block size is given or 2. The #ofSNPs < Min(block_size,iid_count) #similar code elsewhere
            if self.block_size is None or (self.train.sid_count <= self.block_size or self.train.sid_count <= self.train.iid_count+self.test.iid_count):
                train_snps = self.train.read(dtype=dtype).standardize(self.standardizer)
                test_snps = self.test.read(dtype=dtype).standardize(self.standardizer)
                if order == 'F': #numpy's 'dot' always returns 'C' order
                    k_val = test_snps.val.dot(train_snps.val.T).T
                else:
                    k_val = train_snps.val.dot(test_snps.val.T)
                return k_val
            else: #Do in blocks
                #Set the default order to 'C' because with kernels any order is fine and the Python .dot method likes 'C' best.
                if order=='A':
                    order = 'C'
                k_val = np.zeros([self.train.iid_count,self.test.iid_count],dtype=dtype,order=order)
                ct = 0
                ts = time.time()

                for start in range(0, self.train.sid_count, self.block_size):
                    ct += self.block_size
                    train_snps = self.train[:,start:start+self.block_size].read(dtype=dtype).standardize(self.standardizer)
                    test_snps =  self.test [:,start:start+self.block_size].read(dtype=dtype).standardize(self.standardizer)
                    if order == 'F': #numpy's 'dot' always returns 'C' order
                        k_val += test_snps.val.dot(train_snps.val.T).T
                    else:
                        k_val += train_snps.val.dot(test_snps.val.T)
                    if ct % self.block_size==0:
                        diff = time.time()-ts
                        if diff > 1: logging.info("read %s SNPs in %.2f seconds" % (ct, diff))
                return k_val
        else:
            raise Exception("_SnpTrainTest currently only has code for reading all of train x test")


    def __repr__(self):
        s = "_SnpTrainTest(train={0},test={1},standardizer={2}".format(self.train,self.test,self.standardizer)
        if self.block_size is not None:
            s += ",block_size={0}".format(self.block_size)
        s += ")"
        return s

    def copyinputs(self, copier):
        #Doesn't need run_once
        copier.input(self.train)
        copier.input(self.test)
        copier.input(self.standardizer)

def _snps_fixup(snp_input, iid_if_none=None,count_A1=None):
    from pysnptools.snpreader import _snps_fixup as pst_snps_fixup
    return pst_snps_fixup(snp_input,iid_if_none,count_A1)

def _pheno_fixup(pheno_input, iid_if_none=None, missing ='NaN',count_A1=None):

    try:
        ret = Pheno(pheno_input, iid_if_none, missing=missing)
        ret.iid #doing this just to force file load
        return ret
    except:
        return _snps_fixup(pheno_input, iid_if_none=iid_if_none,count_A1=count_A1)

def _kernel_fixup(input, iid_if_none, standardizer, test=None, test_iid_if_none=None, block_size=None, train_snps=None, count_A1=None):
    if test is not None and input is None:
        input = test
        test = None

    if isinstance(input, str) and input.endswith(".npz"):
        return KernelNpz(input)

    if isinstance(input, str):
        input = Bed(input, count_A1=count_A1)     #Note that we don't return here. Processing continues
    if isinstance(test, str):
        test = Bed(test, count_A1=count_A1)      #Note that we don't return here. Processing continues

    if isinstance(input,SnpReader):
        if test is not None:
            return _SnpWholeTest(train=train_snps,test=test,standardizer=standardizer,block_size=block_size)
        else:
            return SnpKernel(input,standardizer=standardizer, block_size=block_size)
        
            
    if input is None:
        return KernelIdentity(iid=iid_if_none,test=test_iid_if_none)

    return input


class FastLMM(object):
    '''
    A predictor, somewhat in the style of scikit-learn, for learning and predicting with linear mixed models.

    **Constructor:**
        :Parameters: * **GB_goal** (int) -- gigabytes of memory the run should use, optional. If not given, will read the test_snps in blocks the same size as the kernel, which is memory efficient with little overhead on computation time.
                     * **force_full_rank** (bool) -- Even if kernels are defined with fewer SNPs than IIDs, create an explicit iid_count x iid_count kernel. Cannot be True if force_low_rank is True.
                     * **force_low_rank** (bool) -- Even if kernels are defined with fewer IIDs than SNPs, create a low-rank iid_count x sid_count kernel. Cannot be True if force_full_rank is True.
                     * **snp_standardizer** (:class:`Standardizer`) -- The PySnpTools standardizer to be apply to SNP data. Choices include :class:`Standardizer.Unit` (Default. Makes values for each SNP have mean zero and standard deviation 1.0, then fills missing with zero) and :class:`Standardizer.Identity` (Do nothing)
                     * **covariate_standardizer** (:class:`Standardizer`) -- The PySnpTools standardizer to be apply to X, the covariate data. Some choices include :class:`Standardizer.Unit` (Default. Fills missing with zero) and :class:`Standardizer.Identity` (do nothing)
                     * **kernel_standardizer** (:class:`KernelStandardizer`) -- The PySnpTools kernel standardizer to be apply to the kernels. Some choices include :class:`KernelStandardizer.DiagKToN` (Default. Make the diagonal sum to iid_count)  and :class:`KernelStandardizer.Identity` (Do nothing)

        :Example:

        >>> import numpy as np
        >>> import logging
        >>> from pysnptools.snpreader import Bed, Pheno
        >>> from fastlmm.util import example_file # Download and return local file name
        >>> from fastlmm.inference import FastLMM
        >>> logging.basicConfig(level=logging.INFO)
        >>> snpreader = Bed(example_file("fastlmm/feature_selection/examples/toydata.5chrom.*","*.bed"),count_A1=False)
        >>> cov_fn = example_file("fastlmm/feature_selection/examples/toydata.cov")
        >>> pheno_fn = example_file("fastlmm/feature_selection/examples/toydata.phe")
        >>> train_idx = np.r_[10:snpreader.iid_count] # iids 10 and on
        >>> test_idx  = np.r_[0:10] # the first 10 iids
        >>> fastlmm = FastLMM(GB_goal=2)
        >>> #We give it phenotype and covariate information for extra examples, but it reorders and intersects the examples, so only training examples are used. 
        >>> _ = fastlmm.fit(K0_train=snpreader[train_idx,:],X=cov_fn,y=pheno_fn) 
        >>> mean, covariance = fastlmm.predict(K0_whole_test=snpreader[test_idx,:],X=cov_fn,count_A1=False)
        >>> print(list(mean.iid[0]), round(mean.val[0,0],7), round(covariance.val[0,0],7))
        ['per0', 'per0'] 0.1791958 0.8995209
        >>> nll = fastlmm.score(K0_whole_test=snpreader[test_idx,:],X=cov_fn,y=pheno_fn,count_A1=False)
        >>> print(round(nll,7))
        13.4623234


        '''

    def __init__(self, GB_goal=None, force_full_rank=False, force_low_rank=False, snp_standardizer=Unit(), covariate_standardizer=Unit(), kernel_standardizer=DiagKtoN()):
        self.GB_goal = GB_goal
        self.force_full_rank = force_full_rank
        self.force_low_rank = force_low_rank
        self.snp_standardizer = snp_standardizer
        self.covariate_standardizer = covariate_standardizer
        self.kernel_standardizer = kernel_standardizer
        self.is_fitted = False

    #!!!update doc to explain h2raw w.r.t h2
    def fit(self, X=None, y=None, K0_train=None, K1_train=None, h2raw=None, mixing=None,count_A1=None):#!!!is this h2 or h2corr????
        """
        Method for training a :class:`FastLMM` predictor. If the examples in X, y, K0_train, K1_train are not the same, they will be reordered and intersected.

        :param X: training covariate information, optional: 
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type X: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__
          (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param y: training phenotype:
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type y: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ 
          (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param K0_train: A similarity matrix or SNPs from which to construct such a similarity matrix.
               Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__.
               If you give a string, can be the name of a PLINK-formated Bed file.
               Can be PySnpTools `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__.
               If you give a string it can be the name of a `KernelNpz <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelnpz>`__ file.
        :type K0_train: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or
               `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param K1_train: A second similarity matrix or SNPs from which to construct such a second similarity matrix. (Also, see 'mixing').
               Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__. If you give a string, can be the name of a PLINK-formated Bed file.
               Can be PySnpTools `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__.
               If you give a string it can be the name of a `KernelNpz <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelnpz>`__ file.
        :type K1_train: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or
               `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param h2raw: A parameter to LMM learning that tells how much weight to give the K's vs. the identity matrix, optional 
                If not given will search for best value.
                If mixing is unspecified, then h2 must also be unspecified.
        :type h2raw: number

        :param mixing: Weight between 0.0 (inclusive, default) and 1.0 (inclusive) given to K1_train relative to K0_train.
                If you give no mixing number and a K1_train is given, the best weight will be learned.
        :type mixing: number

        :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
             alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
        :type count_A1: bool

        :rtype: self, the fitted FastLMM predictor
        """
        self.is_fitted = True
        # should this have a cache file like 'single_snp'?
        #!!!later what happens if missing values in pheno_train?
        #!!!later add code so that X, y, etc can be array-like objects without iid information. In that case, make up iid info

        assert y is not None, "y must be given"

        y = _pheno_fixup(y,count_A1=count_A1)
        assert y.sid_count == 1, "Expect y to be just one variable"
        X = _pheno_fixup(X, iid_if_none=y.iid,count_A1=count_A1)

        K0_train = _kernel_fixup(K0_train, iid_if_none=y.iid, standardizer=self.snp_standardizer,count_A1=count_A1)
        K1_train = _kernel_fixup(K1_train, iid_if_none=y.iid, standardizer=self.snp_standardizer,count_A1=count_A1)

        K0_train, K1_train, X, y = intersect_apply([K0_train, K1_train, X, y],intersect_before_standardize=True) #!!! test this on both K's as None
        from fastlmm.association.single_snp import _set_block_size
        K0_train, K1_train, block_size = _set_block_size(K0_train, K1_train, mixing, self.GB_goal, self.force_full_rank, self.force_low_rank)

        X = X.read()
        # If possible, unit standardize train and test together. If that is not possible, unit standardize only train and later apply
        # the same linear transformation to test. Unit standardization is necessary for FastLMM to work correctly.
        #!!!later is the calculation of the training data's stats done twice???
        X, covar_unit_trained = X.standardize(self.covariate_standardizer,block_size=block_size,return_trained=True) #This also fills missing with the mean

        # add a column of 1's to cov to increase DOF of model (and accuracy) by allowing a constant offset
        X = SnpData(iid=X.iid,
                                sid=self._new_snp_name(X),
                                val=np.c_[X.val,np.ones((X.iid_count,1))],
                                name ="covariate_train w/ 1's")

        y0 =  y.read().val #!!!later would view_ok=True,order='A' be ok because this code already did a fresh read to look for any missing values 

        from fastlmm.association.single_snp import _Mixer #!!!move _combine_the_best_way to another file (e.g. this one)
        K_train, h2raw, mixer = _Mixer.combine_the_best_way(K0_train,K1_train,X.val,y0,mixing,h2raw,force_full_rank=self.force_full_rank,force_low_rank=self.force_low_rank,kernel_standardizer=self.kernel_standardizer,block_size=block_size)

        # do final prediction using lmm.py
        lmm = LMM()

        #Special case: The K kernel is defined implicitly with SNP data
        if mixer.do_g:
            assert isinstance(K_train.standardizer,StandardizerIdentity), "Expect Identity standardizer"
            G_train = K_train.snpreader
            lmm.setG(G0=K_train.snpreader.val)
        else:
            lmm.setK(K0=K_train.val)

        lmm.setX(X.val)
        lmm.sety(y0[:,0])

        # Find the best h2 and also on covariates (not given from new model)
        if h2raw is None:
            res = lmm.findH2() #!!!why is REML true in the return???
        else:
            res = lmm.nLLeval(h2=h2raw)


        #We compute sigma2 instead of using res['sigma2'] because res['sigma2'] is only the pure noise.
        full_sigma2 = float(sum((np.dot(X.val,res['beta']).reshape(-1,1)-y0)**2))/y.iid_count #!!! this is non REML. Is that right?

        ###### all references to 'fastlmm_model' should be here so that we don't forget any
        self.block_size = block_size
        self.beta = res['beta']
        self.h2raw = res['h2']
        self.sigma2 = full_sigma2
        self.U = lmm.U
        self.S = lmm.S
        self.K = lmm.K
        self.G = lmm.G
        self.y = lmm.y
        self.Uy = lmm.Uy
        self.X = lmm.X
        self.UX = lmm.UX
        self.mixer = mixer
        self.covar_unit_trained = covar_unit_trained
        self.K_train_iid = K_train.iid
        self.covar_sid = X.sid
        self.pheno_sid = y.sid
        self.G0_train = K0_train.snpreader if isinstance(K0_train,SnpKernel) else None #!!!later expensive?
        self.G1_train = K1_train.snpreader if isinstance(K1_train,SnpKernel) else None #!!!later expensive?
        return self

    @staticmethod
    def _new_snp_name(snpreader):
        new_snp = "always1"
        while True:
            if not new_snp in snpreader.sid:
                return np.r_[snpreader.sid,[new_snp]]
            new_snp += "_"
    

    def score(self, X=None, y=None, K0_whole_test=None, K1_whole_test=None, iid_if_none=None, return_mse_too=False, return_per_iid=False, count_A1=None):
        """
        Method for calculating the negative log likelihood of testing examples.
        If the examples in X,y,  K0_whole_test, K1_whole_test are not the same, they will be reordered and intersected.

        :param X: testing covariate information, optional: 
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type X: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param y: testing phenotype:
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type y: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param K0_whole_test: A similarity matrix from all the examples to the test examples. Alternatively,
               the test SNPs needed to construct such a similarity matrix.
               Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__. If you give a string, can be the name of a PLINK-formated Bed file.
               Can be PySnpTools `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__. If you give a string it can be the name of a `KernelNpz <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelnpz>`__ file.
        :type K0_whole_test: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param K1_whole_test: A second similarity matrix from all the examples to the test examples. Alternatively,
               the test SNPs needed to construct such a similarity matrix.
               Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__. If you give a string, can be the name of a PLINK-formated Bed file.
               Can be PySnpTools `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__. If you give a string it can be the name of a `KernelNpz <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelnpz>`__ file.
        :type K1_whole_test: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param iid_if_none: Examples to predict for if no X, K0_whole_test, K1_whole_test is provided.
        :type iid_if_none: an ndarray of two strings

        :param return_mse_too: If true, will also return the mean squared error.
        :type return_mse_too: bool

        :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
             alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
        :type count_A1: bool

        :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
             alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
        :type count_A1: bool

        :rtype: a float of the negative log likelihood and, optionally, a float of the mean squared error.
        """
        mean0, covar0 = self.predict(K0_whole_test=K0_whole_test,K1_whole_test=K1_whole_test,X=X,iid_if_none=iid_if_none,count_A1=count_A1)
        y = _pheno_fixup(y, iid_if_none=covar0.iid,count_A1=count_A1)
        mean, covar, y = intersect_apply([mean0, covar0, y])
        mean = mean.read(order='A',view_ok=True).val
        covar = covar.read(order='A',view_ok=True).val
        y_actual = y.read().val
        if not return_per_iid:
            var = multivariate_normal(mean=mean.reshape(-1), cov=covar)
            nll = -np.log(var.pdf(y_actual.reshape(-1)))
            if not return_mse_too:
                return nll
            else:
                mse = ((y_actual-mean)**2).sum()
                return nll, mse
        else:
            if not return_mse_too:
                result = SnpData(iid=y.iid,sid=['nLL'],val=np.empty((y.iid_count,1)),name="nLL")
                for iid_index in range(y.iid_count):
                    var = multivariate_normal(mean=mean[iid_index], cov=covar[iid_index,iid_index])
                    nll = -np.log(var.pdf(y_actual[iid_index]))
                    result.val[iid_index,0] = nll
                return result
            else:
               raise Exception("need code for mse_too")                                  


    def _extract_fixup(kernel):
        assert kernel.iid0_count >= kernel.iid1_count, "Expect iid0 to be at least as long as iid1"


    def predict(self,X=None,K0_whole_test=None,K1_whole_test=None,iid_if_none=None, count_A1=None):
        """
        Method for predicting from a fitted :class:`FastLMM` predictor.
        If the examples in X, K0_whole_test, K1_whole_test are not the same, they will be reordered and intersected.

        :param X: testing covariate information, optional: 
          If you give a string, it should be the file name of a PLINK phenotype-formatted file.
        :type X: a PySnpTools `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ (such as `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__) or string.

        :param K0_whole_test: A similarity matrix from all the examples to the test examples. Alternatively,
               the test SNPs needed to construct such a similarity matrix.
               Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__. If you give a string, can be the name of a PLINK-formated Bed file.
               Can be PySnpTools `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__. If you give a string it can be the name of a `KernelNpz <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelnpz>`__ file.
        :type K0_whole_test: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param K1_whole_test: A second similarity matrix from all the examples to the test examples. Alternatively,
               the test SNPs needed to construct such a similarity matrix.
               Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__. If you give a string, can be the name of a PLINK-formated Bed file.
               Can be PySnpTools `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__. If you give a string it can be the name of a `KernelNpz <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelnpz>`__ file.
        :type K1_whole_test: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__

        :param iid_if_none: Examples to predict for if no X, K0_whole_test, K1_whole_test is provided.
        :type iid_if_none: an ndarray of two strings

        :rtype: A `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__ of the means and a :class:`KernelData` of the covariance
        """

        assert self.is_fitted, "Can only predict after predictor has been fitted"
        #assert K0_whole_test is not None, "K0_whole_test must be given"
        #!!!later is it too wasteful to keep both G0_train, G1_train, and lmm.G when storing to disk?
        #!!!later all _kernel_fixup's should use block_size input

        K0_whole_test_b = _kernel_fixup(K0_whole_test, train_snps=self.G0_train, iid_if_none=iid_if_none, standardizer=self.mixer.snp_trained0, test=K0_whole_test, test_iid_if_none=None, block_size=self.block_size,count_A1=count_A1)
        K1_whole_test = _kernel_fixup(K1_whole_test, train_snps=self.G1_train, iid_if_none=K0_whole_test_b.iid0, standardizer=self.mixer.snp_trained1, test=K1_whole_test, test_iid_if_none=K0_whole_test_b.iid1, block_size=self.block_size,count_A1=count_A1)
        X = _pheno_fixup(X,iid_if_none=K0_whole_test_b.iid1,count_A1=count_A1)
        K0_whole_test_c, K1_whole_test, X = intersect_apply([K0_whole_test_b, K1_whole_test, X],intersect_before_standardize=True,is_test=True)
        X = X.read().standardize(self.covar_unit_trained)
        # add a column of 1's to cov to increase DOF of model (and accuracy) by allowing a constant offset
        X = SnpData(iid=X.iid,
                              sid=self._new_snp_name(X),
                              val=np.c_[X.read().val,np.ones((X.iid_count,1))])
        assert np.array_equal(X.sid,self.covar_sid), "Expect covar sids to be the same in train and test."

        train_idx0 = K0_whole_test_c.iid0_to_index(self.K_train_iid)
        K0_train_test = K0_whole_test_c[train_idx0,:]
        train_idx1 = K1_whole_test.iid0_to_index(self.K_train_iid)
        K1_train_test = K1_whole_test[train_idx1,:]
        test_idx0 = K0_whole_test_c.iid0_to_index(K0_whole_test_c.iid1)
        K0_test_test = K0_whole_test_c[test_idx0,:]
        if K0_test_test.iid0 is not K0_test_test.iid1:
            raise Exception("real assert")
        test_idx1 = K1_whole_test.iid0_to_index(K0_whole_test_c.iid1)
        K1_test_test = K1_whole_test[test_idx1,:]

        if self.mixer.do_g:
            ###################################################
            # low rank from Rasmussen  eq 2.9 + noise term added to covar
            ###################################################
            Gstar = self.mixer.g_mix(K0_train_test,K1_train_test)
            varg = self.h2raw * self.sigma2
            vare = (1.-self.h2raw) * self.sigma2
            Ainv = LA.inv((1./vare) * np.dot(self.G.T,self.G) + (1./varg)*np.eye(self.G.shape[1]))
            testAinv = np.dot(Gstar.test.val, Ainv)
            pheno_predicted = np.dot(X.val,self.beta) + (1./vare) * np.dot(np.dot(testAinv,self.G.T),self.y-np.dot(self.X,self.beta))
            pheno_predicted = pheno_predicted.reshape(-1,1)
            covar  = np.dot(testAinv,Gstar.test.val.T) + vare * np.eye(Gstar.test.val.shape[0])

        else:
            lmm = LMM()
            lmm.U = self.U
            lmm.S = self.S
            lmm.G = self.G
            lmm.y = self.y
            lmm.Uy = self.Uy
            lmm.X = self.X
            lmm.UX = self.UX

            Kstar = self.mixer.k_mix(K0_train_test,K1_train_test) #!!!later do we need/want reads here? how about view_OK?
            lmm.setTestData(Xstar=X.val, K0star=Kstar.val.T)

            Kstar_star = self.mixer.k_mix(K0_test_test,K1_test_test) #!!!later do we need/want reads here?how about view_OK?
            pheno_predicted, covar = lmm.predict_mean_and_variance(beta=self.beta, h2=self.h2raw,sigma2=self.sigma2, Kstar_star=Kstar_star.val)

        #pheno_predicted = lmm.predictMean(beta=self.beta, h2=self.h2,scale=self.sigma2).reshape(-1,1)
        ret0 = SnpData(iid = X.iid, sid=self.pheno_sid,val=pheno_predicted,pos=np.array([[np.nan,np.nan,np.nan]]),name="lmm Prediction")

        from pysnptools.kernelreader import KernelData
        ret1 = KernelData(iid=K0_test_test.iid,val=covar)
        return ret0, ret1

if __name__ == "__main__":
    if True:
        from fastlmm.util import example_file # Download and return local file name
        snpreader = Bed(example_file("fastlmm/feature_selection/examples/toydata.5chrom.*","*.bed"),count_A1=False)

    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
