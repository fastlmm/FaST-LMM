from __future__ import absolute_import
from __future__ import print_function
from pysnptools.util.mapreduce1.runner import *
import logging
import fastlmm.pyplink.plink as plink
from pysnptools.snpreader import Pheno
import pysnptools.util as pstutil
import numpy as np
import scipy.stats as stats
from pysnptools.snpreader import Bed
from fastlmm.util.pickle_io import load, save
import time
import pandas as pd
from fastlmm.inference.lmm_cov import LMM as lmm_cov
import warnings
from pysnptools.snpreader import SnpReader
from pysnptools.snpreader import SnpData
from pysnptools.standardizer import Unit
from pysnptools.standardizer import Identity as SS_Identity
from pysnptools.standardizer import DiagKtoN
from pysnptools.standardizer import UnitTrained
from pysnptools.kernelreader import Identity as KernelIdentity
from pysnptools.kernelreader import KernelData
from pysnptools.kernelreader import SnpKernel
from pysnptools.kernelreader import KernelNpz
from pysnptools.util.mapreduce1 import map_reduce
from pysnptools.util import create_directory_if_necessary
from pysnptools.util.intrangeset import IntRangeSet
from fastlmm.inference.fastlmm_predictor import _snps_fixup, _pheno_fixup, _kernel_fixup, _SnpTrainTest
import fastlmm.inference.linear_regression as lin_reg
from six.moves import range

def single_snp(test_snps, pheno, K0=None,#!!!LATER add warning here (and elsewhere) K0 or K1.sid_count < test_snps.sid_count, might be a covar mix up.(but only if a SnpKernel
                 K1=None, mixing=None,
                 covar=None, covar_by_chrom=None, leave_out_one_chrom=True, output_file_name=None, h2=None, log_delta=None,
                 cache_file=None, GB_goal=None, interact_with_snp=None, force_full_rank=False, force_low_rank=False, G0=None, G1=None, runner=None,
                 count_A1=None):
    """
    Function performing single SNP GWAS using cross validation over the chromosomes and REML. Will reorder and intersect IIDs as needed.
    (For backwards compatibility, you may use 'leave_out_one_chrom=False' to skip cross validation, but that is not recommended.)

    :param test_snps: SNPs to test. Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_. 
           If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type test_snps: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string

    :param pheno: A single phenotype: Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_, for example,
           `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`_ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`_.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           Any IIDs with missing values will be removed.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type pheno: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string

    :param K0: similarity matrix or SNPs from which to create a similarity matrix. If not given, will use test_snps.
           Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_. 
           If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
           If leave_out_one_chrom is True, can be a dictionary from chromosome number to any `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`_
           or the name a `KernelNpz <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelnpz>`_-formated file.
           If leave_out_one_chrom is False, can be any `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`_ or
           name of a `KernelNpz <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelnpz>`_-formated file.
    :type K0: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string
           or dictionary or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`_)

    :param K1: second similarity matrix or SNPs from which to create a second similarity matrix, optional. (Also, see 'mixing').
    :type K1: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string
           or dictionary or `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`_.

    :param mixing: Weight between 0.0 (inclusive, default) and 1.0 (inclusive) given to K1 relative to K0.
            If you give no mixing number and a K1 is given, the best weight will be learned.
    :type mixing: number

    :param covar: covariate information, optional: Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_, for example, `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`_ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`_.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header') #!!!LATER raise error if covar has NaN
    :type covar: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string

    :param covar_by_chrom: dictionary from chromosome number to covariate information, optional:
           The covariate information
           can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_.
           If given, leave_out_one_chrom must be True.
           Both covar and covar_by_chrom can be given.
    :type covar_by_chrom: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string

    :param leave_out_one_chrom: Perform single SNP GWAS via cross validation over the chromosomes. Default to True.
           (Warning: setting False can cause proximal contamination.)
    :type leave_out_one_chrom: boolean    

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created. The output format is tab-delimited text.
    :type output_file_name: file name

    :param h2: A parameter to LMM learning, optional
            If not given will search for best value.
            If mixing is unspecified, then h2 must also be unspecified.
    :type h2: number

    :param log_delta: a re-parameterization of h2 provided for backwards compatibility. h2 is 1./(exp(log_delta)+1)
    :type log_delta: number

    :param cache_file: Name of  file to read or write cached precomputation values to, optional.
                If not given, no cache file will be used.
                If given and file does not exist, will write precomputation values to file.
                If given and file does exist, will read precomputation values from file.
                The file contains the U and S matrix from the decomposition of the training matrix. It is in Python's np.savez (\*.npz) format.
                Calls using the same cache file should have the same 'K0' and 'K1'
                If given and the file does exist then K0 and K1 need not be given.
    :type cache_file: file name

    :param GB_goal: gigabytes of memory the run should use, optional. If not given, will read the test_snps in blocks the same size as the kernel,
        which is memory efficient with little overhead on computation time.
    :type GB_goal: number

    :param interact_with_snp: index of a covariate to perform an interaction test with. 
            Allows for interaction testing (interact_with_snp x snp will be tested)
            default: None

    :param force_full_rank: Even if kernels are defined with fewer SNPs than IIDs, create an explicit iid_count x iid_count kernel. Cannot be True if force_low_rank is True.
    :type force_full_rank: Boolean

    :param force_low_rank: Even if kernels are defined with fewer IIDs than SNPs, create a low-rank iid_count x sid_count kernel. Cannot be True if force_full_rank is True.
    :type force_low_rank: Boolean

    :param G0: Same as K0. Provided for backwards compatibility. Cannot be given if K0 is given.
    :type G0: Same as K0.

    :param G1: Same as K1. Provided for backwards compatibility. Cannot be given if K1 is given.
    :type G1: Same as K1.

    :param runner: a `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`_, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`_

    :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
         alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
    :type count_A1: bool

    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"

    :Example:

    >>> import logging
    >>> from fastlmm.association import single_snp
    >>> from fastlmm.util import example_file # Download and return local file name
    >>> from pysnptools.snpreader import Bed
    >>> logging.basicConfig(level=logging.INFO)
    >>> pheno_fn = example_file("fastlmm/feature_selection/examples/toydata.phe")
    >>> test_snps = example_file("fastlmm/feature_selection/examples/toydata.5chrom.*","*.bed")
    >>> results_dataframe = single_snp(test_snps=test_snps, pheno=pheno_fn, count_A1=False)
    >>> print(results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe))
    null_576 1e-07 10000


    """
    t0 = time.time()
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

    assert test_snps is not None, "test_snps must be given as input"
    test_snps = _snps_fixup(test_snps, count_A1=count_A1)
    pheno = _pheno_fixup(pheno, count_A1=count_A1).read()
    assert pheno.sid_count == 1, "Expect pheno to be just one variable"
    pheno = pheno[(pheno.val==pheno.val)[:,0],:]
    covar = _pheno_fixup(covar, iid_if_none=pheno.iid, count_A1=count_A1)

    if not leave_out_one_chrom:
        assert covar_by_chrom is None, "When 'leave_out_one_chrom' is False, 'covar_by_chrom' must be None"#!!!LATER document covar_by_chrom
        K0 = _kernel_fixup(K0 or G0 or test_snps, iid_if_none=test_snps.iid, standardizer=Unit(),count_A1=count_A1)
        K1 = _kernel_fixup(K1 or G1, iid_if_none=test_snps.iid, standardizer=Unit(),count_A1=count_A1)
        K0, K1, test_snps, pheno, covar  = pstutil.intersect_apply([K0, K1, test_snps, pheno, covar])
        logging.debug("# of iids now {0}".format(K0.iid_count))
        K0, K1, block_size = _set_block_size(K0, K1, mixing, GB_goal, force_full_rank, force_low_rank)

        frame =  _internal_single(K0=K0, test_snps=test_snps, pheno=pheno,
                                    covar=covar, K1=K1,
                                    mixing=mixing, h2=h2, log_delta=log_delta,
                                    cache_file = cache_file, force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                    output_file_name=output_file_name,block_size=block_size, interact_with_snp=interact_with_snp,
                                    runner=runner)
        sid_index_range = IntRangeSet(frame['sid_index'])
        assert sid_index_range == (0,test_snps.sid_count), "Some SNP rows are missing from the output"
    else: 
        chrom_list = list(set(test_snps.pos[:,0])) # find the set of all chroms mentioned in test_snps, the main testing data
        assert not np.isnan(chrom_list).any(), "chrom list should not contain NaN"
        input_files = [test_snps, pheno, covar] + ([] if covar_by_chrom is None else list(covar_by_chrom.values()))
        for Ki in [K0, G0, K1, G1]:
            if isinstance(Ki,dict):
                input_files += list(Ki.values())
            else:
                input_files.append(Ki)

        def nested_closure(chrom):
            test_snps_chrom = test_snps[:,test_snps.pos[:,0]==chrom]
            covar_chrom = _create_covar_chrom(covar, covar_by_chrom, chrom)
            cache_file_chrom = None if cache_file is None else cache_file + ".{0}".format(chrom)

            K0_chrom = _K_per_chrom(K0 or G0 or test_snps, chrom, test_snps.iid)
            K1_chrom = _K_per_chrom(K1 or G1, chrom, test_snps.iid)

            K0_chrom, K1_chrom, test_snps_chrom, pheno_chrom, covar_chrom  = pstutil.intersect_apply([K0_chrom, K1_chrom, test_snps_chrom, pheno, covar_chrom])
            logging.debug("# of iids now {0}".format(K0_chrom.iid_count))
            K0_chrom, K1_chrom, block_size = _set_block_size(K0_chrom, K1_chrom, mixing, GB_goal, force_full_rank, force_low_rank)

            distributable = _internal_single(K0=K0_chrom, test_snps=test_snps_chrom, pheno=pheno_chrom,
                                        covar=covar_chrom, K1=K1_chrom,
                                        mixing=mixing, h2=h2, log_delta=log_delta, cache_file=cache_file_chrom,
                                        force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                        output_file_name=None, block_size=block_size, interact_with_snp=interact_with_snp,
                                        runner=Local())
            
            return distributable

        def reducer_closure(frame_sequence):
            frame = pd.concat(frame_sequence)
            frame.sort_values(by="PValue", inplace=True)
            frame.index = np.arange(len(frame))
            if output_file_name is not None:
                frame.to_csv(output_file_name, sep="\t", index=False)
            logging.info("PhenotypeName\t{0}".format(pheno.sid[0]))
            logging.info("SampleSize\t{0}".format(test_snps.iid_count))
            logging.info("SNPCount\t{0}".format(test_snps.sid_count))
            logging.info("Runtime\t{0}".format(time.time()-t0))

            return frame

        frame = map_reduce(chrom_list,
                   mapper = nested_closure,
                   reducer = reducer_closure,
                   input_files = input_files,
                   output_files = [output_file_name],
                   name = "single_snp (leave_out_one_chrom), out='{0}'".format(output_file_name),
                   runner = runner)

    return frame

overhead_gig = .127
factor = 8.5  # found via trial and error

def _GB_goal_from_block_size(block_size, iid_count, kernel_gig):
    left_bytes = block_size * (iid_count * 8.0 *factor)
    left_gig = left_bytes / 1024.0**3
    GB_goal = left_gig + overhead_gig + kernel_gig
    return GB_goal

def _block_size_from_GB_goal(GB_goal, iid_count, min_count):
    kernel_bytes = iid_count * min_count * 8
    kernel_gig = kernel_bytes / (1024.0**3)

    if GB_goal is None:
        GB_goal = _GB_goal_from_block_size(min_count, iid_count, kernel_gig)
        logging.info("Setting GB_goal to {0} GB".format(GB_goal))
        return min_count

    left_gig = GB_goal - overhead_gig - kernel_gig
    if left_gig <= 0:
        warnings.warn("The full kernel and related operations will likely not fit in the goal_memory")
    left_bytes = left_gig * 1024.0**3
    snps_at_once = left_bytes / (iid_count * 8.0 * factor)
    block_size = int(snps_at_once)

    if block_size < min_count:
        block_size = min_count
        GB_goal = _GB_goal_from_block_size(block_size, iid_count, kernel_gig)
        warnings.warn("Can't meet goal_memory without loading too few snps at once. Resetting GB_goal to {0} GB".format(GB_goal))

    return block_size

def single_snp_leave_out_one_chrom(*args, **kwargs):
    """
    .. deprecated:: 0.2.22
       Use :meth:`single_snp` instead.
    
    """
    warnings.warn("'single_snp_leave_out_one_chrom' is deprecated. Use 'single_snp(...) instead.", DeprecationWarning)
    return single_snp(*args, **kwargs)

def _K_per_chrom(K, chrom, iid,count_A1=None):
    if K is None:
        return KernelIdentity(iid)
    else:
        if isinstance(K,dict):
            return _kernel_fixup(K[chrom], iid_if_none=iid, standardizer=Unit(),count_A1=count_A1)
        else:
            K_all = _kernel_fixup(K, iid_if_none=iid, standardizer=Unit(),count_A1=count_A1) 
            if not isinstance(K_all, SnpKernel):
                raise Exception("Don't know how to make '{0}' work per chrom".format(K_all))
            return SnpKernel(K_all.snpreader[:,K_all.pos[:,0] != chrom],K_all.standardizer)
                

#!!!move to own file?
class _Mixer(object):
    def __init__(self, do_g, kernel_trained0,kernel_trained1,mixing):
        self.do_g = do_g
        self.kernel_trained0 = kernel_trained0
        self.kernel_trained1 = kernel_trained1
        self.mixing = mixing
        self.snp_trained0 = None
        self.snp_trained1 = None

    def k_mix(self,K0,K1): #!!!later add special case code for mixing==1 and 0
        K0_b = K0.read().standardize(self.kernel_trained0)
        K1_b = K1.read().standardize(self.kernel_trained1)
        # similar code elsewhere
        K = np.empty(K0_b.val.shape)
        _mix_from_Ks(K, K0_b.val, K1_b.val, self.mixing)
        K = KernelData(val=K,iid0=K0_b.iid0,iid1=K0_b.iid1)
        return K

    # g_mix doesn't care about block_size because if we are low-rank the number SNPs is small.
    def g_mix(self,K0,K1):
        mixing = self.mixing

        if mixing == 1 or isinstance(K0, KernelIdentity):
            assert K1.standardizer is self.snp_trained1, "real assert"
            G_train = K1.train.read().standardize(self.snp_trained1).standardize(self.kernel_trained1)#!!!later this a good place to read?
            G_test = K1.test.read().standardize(self.snp_trained1).standardize(self.kernel_trained1)#!!!later this a good place to read?
            K = _SnpTrainTest(train=G_train,test=G_test,standardizer=SS_Identity(), block_size=None)
            return K

        if mixing == 0 or isinstance(K1, KernelIdentity):
            assert K0.standardizer is self.snp_trained0, "real assert"
            G_train = K0.train.read().standardize(self.snp_trained0).standardize(self.kernel_trained0)#!!!later this a good place to read?
            G_test = K0.test.read().standardize(self.snp_trained0).standardize(self.kernel_trained0)#!!!later this a good place to read?
            K = _SnpTrainTest(train=G_train,test=G_test,standardizer=SS_Identity(), block_size=None)
            return K

        #!!!later why are we processing the training data again????
        assert K0.standardizer is self.snp_trained0, "real assert"
        assert isinstance(K0, _SnpTrainTest), "Expect K0 to be a _SnpTrainTest"
        assert K1.standardizer is self.snp_trained1, "real assert"
        G0_train = K0.train.read().standardize(self.snp_trained0).standardize(self.kernel_trained0)#!!!later this a good place to read?
        G1_train = K1.train.read().standardize(self.snp_trained1).standardize(self.kernel_trained1)#!!!later this a good place to read?
        G0_test = K0.test.read().standardize(self.snp_trained0).standardize(self.kernel_trained0)#!!!later this a good place to read?
        G1_test = K1.test.read().standardize(self.snp_trained1).standardize(self.kernel_trained1)#!!!later this a good place to read?
        G_train = np.empty((K0.iid0_count, K0.train.sid_count + K1.train.sid_count))
        G_test = np.empty((K0.iid1_count, K0.train.sid_count + K1.train.sid_count))
        _mix_from_Gs(G_train, G0_train.val, G1_train.val, self.mixing)
        _mix_from_Gs(G_test, G0_test.val, G1_test.val, self.mixing)
        G_train = SnpData(iid=K0.iid0,
                            sid=np.concatenate((K0.train.sid,K1.train.sid),axis=0),
                            val=G_train,name="{0}&{1}".format(G0_train,G1_train),
                            pos=np.concatenate((K0.train.pos,K1.train.pos),axis=0)
                            )
        G_test = SnpData(iid=K0.iid1,
                            sid=np.concatenate((K0.train.sid,K1.train.sid),axis=0),
                            val=G_test,name="{0}&{1}".format(G0_test,G1_test),
                            pos=np.concatenate((K0.train.pos,K1.train.pos),axis=0)
                            )
        K = _SnpTrainTest(train=G_train,test=G_test,standardizer=SS_Identity(), block_size=None)
        return K

    def to_np(self):
        dict = {'do_g':self.do_g,
                #'G0_trained.stats':self.G0_trained.stats if self.G0_trained is not None else [],
                #'G1_trained.stats':self.G1_trained.stats if self.G1_trained is not None else [],
                'factor0':self.kernel_trained0,
                'factor1':self.kernel_trained1,
                'mixing':self.mixing}
        return dict

    @staticmethod
    def from_np(data):
        mixer= _Mixer(
                    do_g = bool(data['do_g']),
                    kernel_trained0 = float(data['factor0']),
                    kernel_trained1 = float(data['factor1']),
                    mixing = float(data['mixing'])
                    )
        return mixer

    @staticmethod
    def combine_the_best_way(K0, K1, covar, y, mixing, h2, force_full_rank=False, force_low_rank=False,snp_standardizer=None,kernel_standardizer=None,block_size=None):
        from pysnptools.kernelstandardizer import Identity as KS_Identity

        assert K0.iid0 is K0.iid1, "Expect K0 to be square"
        assert K1.iid0 is K1.iid1, "Expect K1 to be square"
        assert K0 is not None
        assert K1 is not None
        assert np.array_equal(K0.iid,K1.iid), "Expect K0 and K1 to having matching iids"
        assert kernel_standardizer is not None, "expect values for kernel_standardizer"

        mixer = _Mixer(False,KS_Identity(),KS_Identity(),mixing)

        sid_count_0 = _Mixer.sid_counter(K0, force_full_rank, force_low_rank)
        sid_count_1 = _Mixer.sid_counter(K1, force_full_rank, force_low_rank)

        #################################
        # Both Identity (or not given)
        #################################
        if sid_count_0 + sid_count_1 == 0:
            h2 = h2 or 0
            mixer.mixing = mixer.mixing or 0
            K = K0.read() #would be nice to use LinearRegression or low-rank with 0 snps

        #################################
        #
        #################################
        elif sid_count_0 + sid_count_1 < K0.iid_count or force_low_rank:
            mixer.do_g = True
            #!!!there is no need for block_size here because we want G0 in full. But if starting with SNPs and not low-rank then batches are needed and the two standardizers must be remembered for use later

            if sid_count_0 > 0:
                K0, mixer.snp_trained0, mixer.kernel_trained0 = K0._read_with_standardizing(to_kerneldata=not mixer.do_g, kernel_standardizer=kernel_standardizer, return_trained=True)
            if sid_count_1 > 0:
                K1, mixer.snp_trained1, mixer.kernel_trained1 = K1._read_with_standardizing(to_kerneldata=not mixer.do_g, kernel_standardizer=kernel_standardizer, return_trained=True)

            if sid_count_1 == 0:
                mixer.mixing = mixer.mixing or 0
                K = K0
            elif sid_count_0 == 0:
                mixer.mixing = mixer.mixing or 1
                K = K1
            else:
                if mixer.do_g:
                    G = np.empty((K0.iid_count, K0.sid_count + K1.sid_count))
                    if mixer.mixing is None:
                        mixer.mixing, h2 = _find_mixing_from_Gs(G, covar, K0.snpreader.val, K1.snpreader.val, h2, y)

                    if mixer.mixing == 0:
                        K = K0
                    elif mixer.mixing == 1:
                        K = K1
                    else:
                        _mix_from_Gs(G, K0.snpreader.val, K1.snpreader.val, mixer.mixing)
                        G = SnpData(iid=K0.iid,
                                            sid=["K0_{0}".format(i) for i in range(K0.sid_count)]+["K1_{0}".format(i) for i in range(K1.sid_count)], #rename the sids so that they can't collide.
                                            val=G,name="{0}&{1}".format(K0.snpreader,K1.snpreader),
                                            pos=np.concatenate((K0.pos,K1.pos),axis=0)
                                            )
                        K = SnpKernel(G,SS_Identity(),block_size=block_size)
        else:
            mixer.do_g = False
            if sid_count_0 > 0: #!!!but what if we have SNP data but still need to remember the standardizer?
                K0, mixer.snp_trained0, mixer.kernel_trained0 = K0._read_with_standardizing(to_kerneldata=True,return_trained=True)#!!!pass in a new argument, the kernel_standardizer(???)

            if sid_count_1 > 0:
                K1, mixer.snp_trained1, mixer.kernel_trained1 = K1._read_with_standardizing(to_kerneldata=True,return_trained=True)

            if sid_count_1 == 0:
                mixer.mixing = mixer.mixing or 0
                K = K0
            elif sid_count_0 == 0:
                mixer.mixing = mixer.mixing or 1
                K = K1
            else:
                K = np.empty(K0.val.shape)
                if mixer.mixing is None:
                    mixer.mixing, h2 = _find_mixing_from_Ks(K, covar, K0.val, K1.val, h2, y)
                _mix_from_Ks(K, K0.val, K1.val, mixer.mixing)
                assert K.shape[0] == K.shape[1] and abs(np.diag(K).sum() - K.shape[0]) < 1e-7, "Expect mixed K to be standardized"
                K = KernelData(val=K,iid=K0.iid)

        return K, h2, mixer

    @staticmethod
    def sid_counter(K, force_full_rank, force_low_rank):
        if isinstance(K,KernelIdentity):
            return 0
        if force_full_rank:
            return np.inf
        if isinstance(K,SnpKernel):
            return K.sid_count
        else:
            return np.inf

def _set_block_size(K0, K1, mixing, GB_goal, force_full_rank, force_low_rank):
    min_count = _internal_determine_block_size(K0, K1, mixing, force_full_rank, force_low_rank)
    iid_count = K0.iid_count if K0 is not None else K1.iid_count
    block_size = _block_size_from_GB_goal(GB_goal, iid_count, min_count)
    #logging.info("Dividing SNPs by {0}".format(-(test_snps.sid_count//-block_size)))

    try:
        K0.block_size = block_size
    except:
        pass # ignore

    try:
        K1.block_size = block_size
    except:
        pass # ignore

    return K0, K1, block_size


def _internal_determine_block_size(K0, K1, mixing, force_full_rank, force_low_rank):
    assert not(force_full_rank and force_low_rank), "real assert"

    if isinstance(K0,SnpKernel) and K0.snpreader.sid_count == 0:
        K0 = KernelIdentity(K0.iid)
    if isinstance(K1,SnpKernel) and K1.snpreader.sid_count == 0:
        K1 = KernelIdentity(K1.iid)


    ##########################
    # A special case: both kernels are the Identity so just return the first one
    ##########################
    if isinstance(K0,KernelIdentity) and isinstance(K1,KernelIdentity):
        return K0.iid_count

    ##########################
    # Special cases: mixing says to use just one kernel or the other kernel is just identity, so just return one kernel
    ##########################
    if mixing == 0.0 or isinstance(K1,KernelIdentity):
        if isinstance(K0,SnpKernel) and not force_full_rank and (force_low_rank or K0.snpreader.sid_count < K0.iid_count):
            return K0.snpreader.sid_count
        else:
            return K0.iid_count

    if mixing == 1.0 or isinstance(K0,KernelIdentity):
        if isinstance(K1,SnpKernel) and not force_full_rank and (force_low_rank or K1.snpreader.sid_count < K1.iid_count):
            return K1.snpreader.sid_count
        else:
            return K1.iid_count

    ##########################
    # A special case: Treat the kernels as collections of snps (i.e. low-rank)
    ##########################
    if (isinstance(K0,SnpKernel) and isinstance(K1,SnpKernel) and not force_full_rank
        and (force_low_rank or K0.snpreader.sid_count + K1.snpreader.sid_count < K0.iid_count)):
        return K0.snpreader.sid_count + K1.snpreader.sid_count 

    ##########################
    # The most general case, treat the new kernels as kernels (i.e.. full rank)
    ##########################
    return K0.iid_count

def _create_dataframe(row_count):
    dataframe = pd.DataFrame(
        index=np.arange(row_count),
        columns=('sid_index', 'SNP', 'Chr', 'GenDist', 'ChrPos', 'PValue', 'SnpWeight', 'SnpWeightSE','SnpFractVarExpl','Mixing', 'Nullh2')
        )
    #!!Is this the only way to set types in a dataframe?
    dataframe['sid_index'] = dataframe['sid_index'].astype(np.float)
    dataframe['Chr'] = dataframe['Chr'].astype(np.float)
    dataframe['GenDist'] = dataframe['GenDist'].astype(np.float)
    dataframe['ChrPos'] = dataframe['ChrPos'].astype(np.float)
    dataframe['PValue'] = dataframe['PValue'].astype(np.float)
    dataframe['SnpWeight'] = dataframe['SnpWeight'].astype(np.float)
    dataframe['SnpWeightSE'] = dataframe['SnpWeightSE'].astype(np.float)
    dataframe['SnpFractVarExpl'] = dataframe['SnpFractVarExpl'].astype(np.float)
    dataframe['Mixing'] = dataframe['Mixing'].astype(np.float)
    dataframe['Nullh2'] = dataframe['Nullh2'].astype(np.float)

    return dataframe


def _internal_single(K0, test_snps, pheno, covar, K1,
                 mixing, h2, log_delta,
                 cache_file, force_full_rank, force_low_rank,
                 output_file_name, block_size, interact_with_snp, runner):
    assert K0 is not None, "real assert"
    assert K1 is not None, "real assert"
    assert block_size is not None, "real assert"
    assert mixing is None or 0.0 <= mixing <= 1.0
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

    assert h2 is None or log_delta is None, "if h2 is specified, log_delta may not be specified"
    if log_delta is not None:
        h2 = 1.0/(np.exp(log_delta)+1)

    covar = np.c_[covar.read(view_ok=True,order='A').val,np.ones((test_snps.iid_count, 1))]  #view_ok because np.c_ will allocation new memory

    y =  pheno.read(view_ok=True,order='A').val #view_ok because this code already did a fresh read to look for any missing values 

    if cache_file is not None and os.path.exists(cache_file):
        lmm = lmm_cov(X=covar, Y=y, G=None, K=None)
        with np.load(cache_file) as data: #!! similar code in epistasis
            lmm.U = data['arr_0']
            lmm.S = data['arr_1']
            h2 = data['arr_2'][0]
            mixing = data['arr_2'][1]
    else:
        K, h2, mixer = _Mixer.combine_the_best_way(K0, K1, covar, y, mixing, h2, force_full_rank=force_full_rank, force_low_rank=force_low_rank,kernel_standardizer=DiagKtoN())
        mixing = mixer.mixing

        if mixer.do_g:
            lmm = lmm_cov(X=covar, Y=y, K=None, G=K.snpreader.val, inplace=True)
        else:
            #print(covar.sum(),y.sum(),K.val.sum(),covar[0],y[0],K.val[0,0])
            lmm = lmm_cov(X=covar, Y=y, K=K.val, G=None, inplace=True)

        if h2 is None:
            result = lmm.findH2()
            h2 = result['h2']
        logging.info("h2={0}".format(h2))

        if cache_file is not None and not os.path.exists(cache_file):
            pstutil.create_directory_if_necessary(cache_file)
            lmm.getSU()
            np.savez(cache_file, lmm.U,lmm.S,np.array([h2,mixing])) #using np.savez instead of pickle because it seems to be faster to read and write

    if interact_with_snp is not None:
        logging.info("interaction with %i" % interact_with_snp)
        assert 0 <= interact_with_snp < covar.shape[1]-1, "interact_with_snp is out of range"
        interact = covar[:,interact_with_snp].copy()
        interact -=interact.mean()
        interact /= interact.std()
    else:
        interact = None

    work_count = -(test_snps.sid_count // -block_size) #Find the work count based on batch size (rounding up)

    # We define three closures, that is, functions define inside function so that the inner function has access to the local variables of the outer function.
    def debatch_closure(work_index):
        return test_snps.sid_count * work_index // work_count

    def mapper_closure(work_index):
        if work_count > 1: logging.info("single_snp: Working on snp block {0} of {1}".format(work_index,work_count))
        do_work_time = time.time()
        start = debatch_closure(work_index)
        end = debatch_closure(work_index+1)

        snps_read = test_snps[:,start:end].read().standardize()
        if interact_with_snp is not None:
            variables_to_test = snps_read.val * interact[:,np.newaxis]
        else:
            variables_to_test = snps_read.val
        res = lmm.nLLeval(h2=h2, dof=None, scale=1.0, penalty=0.0, snps=variables_to_test)

        beta = res['beta']
        
        chi2stats = beta*beta/res['variance_beta']
        #p_values = stats.chi2.sf(chi2stats,1)[:,0]
        assert test_snps.iid_count == lmm.U.shape[0]
        p_values = stats.f.sf(chi2stats,1,lmm.U.shape[0]-(lmm.linreg.D+1))[:,0]#note that G.shape is the number of individuals#

        dataframe = _create_dataframe(snps_read.sid_count)
        dataframe['sid_index'] = np.arange(start,end)
        dataframe['SNP'] = np.array(snps_read.sid, 'str') #This will be ascii on Python2 and unicode on Python3
        dataframe['Chr'] = snps_read.pos[:,0]
        dataframe['GenDist'] = snps_read.pos[:,1]
        dataframe['ChrPos'] = snps_read.pos[:,2] 
        dataframe['PValue'] = p_values
        dataframe['SnpWeight'] = beta[:,0]
        dataframe['SnpWeightSE'] = np.sqrt(res['variance_beta'][:,0])
        dataframe['SnpFractVarExpl'] = np.sqrt(res['fraction_variance_explained_beta'][:,0])
        dataframe['Mixing'] = np.zeros((snps_read.sid_count)) + mixing
        dataframe['Nullh2'] = np.zeros((snps_read.sid_count)) + h2

        logging.info("time={0}".format(time.time()-do_work_time))

        #logging.info(dataframe)
        return dataframe

    def reducer_closure(result_sequence):
        if output_file_name is not None:
            create_directory_if_necessary(output_file_name)

        frame = pd.concat(result_sequence)
        frame.sort_values(by="PValue", inplace=True)
        frame.index = np.arange(len(frame))

        if output_file_name is not None:
            frame.to_csv(output_file_name, sep="\t", index=False)

        return frame

    frame = map_reduce(range(work_count),
                       mapper=mapper_closure,reducer=reducer_closure,
                       input_files=[test_snps],output_files=[output_file_name],
                       name="single_snp(output_file={0})".format(output_file_name),
                       runner=runner)
    return frame



def _create_covar_chrom(covar, covar_by_chrom, chrom,count_A1=None):
    if covar_by_chrom is not None:
        covar_by_chrom_chrom = covar_by_chrom[chrom]
        covar_by_chrom_chrom = _pheno_fixup(covar_by_chrom_chrom, iid_if_none=covar,count_A1=count_A1)
        covar_after,  covar_by_chrom_chrom = pstutil.intersect_apply([covar,  covar_by_chrom_chrom])
        ret = SnpData(iid=covar_after.iid,sid=np.r_[covar_after.sid,covar_by_chrom_chrom.sid],
                      val=np.c_[covar_after.read(order='A',view_ok=True).val,
                                covar_by_chrom_chrom.read(order='A',view_ok=True).val]) #view_ok because np.c_ will allocate new memory.
        return ret
    else:
        return covar


def _find_mixing_from_Gs(G, covar, G0_standardized_val, G1_standardized_val, h2, y):
    logging.info("starting _find_mixing_from_Gs")
    import fastlmm.util.mingrid as mingrid
    assert h2 is None, "if mixing is None, expect h2 to also be None"
    resmin=[None]
    def f(mixing,G0_standardized_val=G0_standardized_val,G1_standardized_val=G1_standardized_val,covar=covar,y=y,**kwargs):

        if not isinstance(mixing, (int, int, float, complex)):
            assert mixing.ndim == 1 and mixing.shape[0] == 1
            mixing = mixing[0]

        _mix_from_Gs(G, G0_standardized_val,G1_standardized_val,mixing)
        lmm = lmm_cov(X=covar, Y=y, G=G, K=None, inplace=True)
        result = lmm.findH2()
        if (resmin[0] is None) or (result['nLL']<resmin[0]['nLL']):
            resmin[0]=result
        logging.info("mixing_from_Gs\t{0}\th2\t{1}\tnLL\t{2}".format(mixing,result['h2'],result['nLL']))
        #logging.info("reporter:counter:single_snp,find_mixing_from_Gs_count,1")
        assert not np.isnan(result['nLL']), "nLL should be a number (not a NaN)"
        return result['nLL']
    mixing,nLL = mingrid.minimize1D(f=f, nGrid=10, minval=0.0, maxval=1.0,verbose=False)

    if not isinstance(mixing, (int, int, float, complex)):
        assert mixing.ndim == 1 and mixing.shape[0] == 1
        mixing = mixing[0]

    h2 = resmin[0]['h2']
    return mixing, h2

def _find_mixing_from_Ks(K, covar, K0_val, K1_val, h2, y):
    logging.info("starting _find_mixing_from_Ks")
    import fastlmm.util.mingrid as mingrid
    assert h2 is None, "if mixing is None, expect h2 to also be None"
    resmin=[None]
    def f(mixing,K0_val=K0_val,K1_val=K1_val,covar=covar,y=y,**kwargs):

        if not isinstance(mixing, (int, int, float, complex)):
            assert mixing.ndim == 1 and mixing.shape[0] == 1
            mixing = mixing[0]

        _mix_from_Ks(K, K0_val,K1_val,mixing)
        lmm = lmm_cov(X=covar, Y=y, G=None, K=K, inplace=True)
        result = lmm.findH2()
        if (resmin[0] is None) or (result['nLL']<resmin[0]['nLL']):
            resmin[0]=result
        logging.debug("mixing_from_Ks\t{0}\th2\t{1}\tnLL\t{2}".format(mixing,result['h2'],result['nLL']))
        #logging.info("reporter:counter:single_snp,find_mixing_from_Ks_count,1")
        assert not np.isnan(result['nLL']), "nLL should be a number (not a NaN)"
        return result['nLL']
    mixing,nLL = mingrid.minimize1D(f=f, nGrid=10, minval=0.0, maxval=1.0,verbose=False)

    if not isinstance(mixing, (int, int, float, complex)):
        assert mixing.ndim == 1 and mixing.shape[0] == 1
        mixing = mixing[0]
    h2 = resmin[0]['h2']
    return mixing, h2


#!!!later move these to _Mixing
def _mix_from_Gs(G, G0_standardized_val, G1_standardized_val, mixing):
    #logging.info("concat G1, mixing {0}".format(mixing))
    G[:,0:G0_standardized_val.shape[1]] = G0_standardized_val
    G[:,0:G0_standardized_val.shape[1]] *= (np.sqrt(1.0-mixing))
    G[:,G0_standardized_val.shape[1]:] = G1_standardized_val
    G[:,G0_standardized_val.shape[1]:] *= np.sqrt(mixing)

def _mix_from_Ks(K, K0_val, K1_val, mixing):
    K[:,:] = K0_val * (1.0-mixing) + K1_val * mixing

if __name__ == "__main__":
    if False:
        import logging
        from fastlmm.association import single_snp
        from fastlmm.util import example_file # Download and return local file name
        from pysnptools.snpreader import Bed
        logging.basicConfig(level=logging.INFO)
        pheno_fn = example_file("fastlmm/feature_selection/examples/toydata.phe")
        test_snps = example_file("fastlmm/feature_selection/examples/toydata.5chrom.*","*.bed")
        results_dataframe = single_snp(test_snps=test_snps, pheno=pheno_fn, count_A1=False)
        print(results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe))



    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()

