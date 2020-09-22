from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import logging
from fastlmm.association import single_snp
from sklearn.model_selection import KFold
import pandas as pd
import os
import time

import pysnptools.util as pstutil
from pysnptools.standardizer import Unit
from pysnptools.kernelreader import KernelReader
from pysnptools.kernelreader import KernelData
from pysnptools.util.mapreduce1 import map_reduce

from fastlmm.inference import FastLMM
from fastlmm.inference.fastlmm_predictor import _snps_fixup, _pheno_fixup, _kernel_fixup
from fastlmm.association.single_snp import _K_per_chrom
from fastlmm.association import single_snp_linreg
from six.moves import range

#!!!move this
class _SnpWholeWithTrain(KernelReader):
    def __init__(self,whole,train_idx, standardizer, block_size):
        self.whole = whole
        self.train_idx = train_idx
        self.standardizer = standardizer
        self.block_size = block_size

    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        #case 1:
        if row_index_or_none is None and col_index_or_none is None:

            #Do all-at-once (not in blocks) if 1. No block size is given or 2. The #ofSNPs < Min(block_size,iid_count) #similar code elsewhere
            if self.block_size is None or (self.whole.sid_count <= self.block_size or self.whole.sid_count <= self.whole.iid_count):
                # read blocks of whole
                whole = self.whole.read(dtype=dtype,force_python_only=force_python_only)
                # train standardizer on complement of test_idx
                train = whole[self.train_idx,:].read(force_python_only=force_python_only)
                _, trained_std = self.standardizer.standardize(train,return_trained=True,force_python_only=force_python_only)
                # apply standardizer to whole
                whole = whole.standardize(trained_std)
                # multiply
                if order == 'F': #numpy's 'dot' always returns 'C' order
                    k_val = np.dot(whole.val,whole.val.T).T
                else:
                    k_val = np.dot(whole.val,whole.val.T)
                return k_val
            else:
                #Set the default order to 'C' because with kernels any order is fine and the Python .dot method likes 'C' best.
                if order=='A':
                    order = 'C'
                k_val = np.zeros([self.whole.iid_count,self.whole.iid_count],dtype=dtype,order=order)
                ct = 0
                ts = time.time()

                for start in range(0, self.whole.sid_count, self.block_size):
                    ct += self.block_size
                    # read blocks of whole
                    whole = self.whole[:,start:start+self.block_size].read(dtype=dtype,force_python_only=force_python_only)
                    # train standardizer on complement of test_idx
                    train = whole[self.train_idx,:].read(force_python_only=force_python_only)
                    _, trained_std = self.standardizer.standardize(train,return_trained=True,force_python_only=force_python_only)
                    # apply standardizer to whole
                    whole = whole.standardize(trained_std)
                    # multiply
                    if order == 'F': #numpy's 'dot' always returns 'C' order
                        k_val += np.dot(whole.val,whole.val.T).T
                    else:
                        k_val += np.dot(whole.val,whole.val.T)
                return k_val
        else:
            raise Exception("need code")

    #!!!To be fully supported class, may need to define more methods

    @property
    def row(self):
        return self.whole.iid

    @property
    def col(self):
        return self.whole.iid

    def __repr__(self):
        s = "_SnpWholeWithTrain(whole={0},train_idx=[...],standardizer={1}".format(self.whole,self.standardizer)
        if self.block_size is not None:
            s += ",block_size={0}".format(self.block_size)
        s += ")"
        return s

    def copyinputs(self, copier):
        #Doesn't need run_once
        copier.input(self.whole)
        copier.input(self.standardizer)


def _nll_plot(k_list,nLL_list):
    import matplotlib.pyplot as plt
    import pylab

    k_list = np.array(k_list)
    nLL_list = np.array(nLL_list)
    is_ok = nLL_list<np.inf
    nLL_list = nLL_list[is_ok]
    k_list = k_list[is_ok]

    pylab.plot(k_list, nLL_list,"-bo")
    plt.xlabel('# of top SNPs')
    plt.ylabel('nLL')
    pylab.xscale("log")
    pylab.show()

def _kfold(iid_count, n_folds, seed, end_with_all=False, iid_to_index=None):
    '''
    When n_folds is 1, then only one thing will be returned, if with end_will_all is True
    If n_folds is a string, will read splits from that string as a file name.
    If n_folds is negative (e.g. -2) then will just give the first fold.
    '''
    if isinstance(n_folds,str):
        def index_list(table):
            result = iid_to_index([[fid]*2 for fid in table.CID])
            return result
        table=pd.read_csv(n_folds,delimiter='\s',comment=None,engine='python')
        fold_count = 1+max(table.Fold)
        result = []
        for fold_index in range(fold_count):
            train = index_list(table[table.Fold!=fold_index])
            test = index_list(table[table.Fold==fold_index])
            result.append((fold_index, [train,test]))
        if end_with_all:
            result.append((fold_count, [list(range(iid_count)),[]]))
        return result
    if n_folds == 1:
        logging.info("Running test-on-train")
        return [(0, [list(range(iid_count)),list(range(iid_count))])]

    if n_folds < 0:
        logging.info("Running just one train/test split")
        result = list(enumerate(KFold(n_splits=-n_folds, random_state=seed, shuffle=True).split(list(range(iid_count)))))[0:1]
        if end_with_all:
            result = result +[(1, [list(range(iid_count)),[]])]
        return result

    result = list(enumerate(KFold(n_splits=n_folds, random_state=seed, shuffle=True).split(list(range(iid_count)))))
    if end_with_all:
        result = result +[(n_folds, [list(range(iid_count)),[]])]
    return result 

#!!!This doesn't (and shouldn't) work when everything is from the same chrom, right? Add a better error message for that case.
def single_snp_all_plus_select(test_snps, pheno, G=None, covar=None,
                 k_list = None,
                 n_folds=10, #1 is special and means test on train
                 seed = 0, output_file_name = None,
                 GB_goal=None, force_full_rank=False, force_low_rank=False, mixing=None, h2=None, do_plot=False, runner=None, count_A1=None):
    """
    Function performing single SNP GWAS based on two kernels. The first kernel is based on all SNPs. The second kernel is a similarity matrix
    constructed of the top *k* SNPs where the SNPs are ordered via the PValue from :meth:`.single_snp` and *k* is determined via out-of-sample prediction.
    All work is done via 'leave_out_one_chrom', that one chromosome is tested and the kernels are constructed from the other chromosomes.
    Will reorder and intersect IIDs as needed.

    :param test_snps: SNPs to test. Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type test_snps: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string

    :param pheno: A single phenotype: Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_, for example, `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`_ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`_.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           Any IIDs with missing values will be removed.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type pheno: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string

    :param G: SNPs from which to create a similarity matrix of the top *k* SNPs. If not given, will use test_snps.
           Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string

    :param covar: covariate information, optional: Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_, for example, `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`_ or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`_.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type covar: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`_ or a string

    :param k_list: Values of *k* (in addition to 0) to test. Default to [1,2,4,8,...8192].
    :type k_list: list of numbers

    :param n_folds: Number of folds of cross validation to use for out-of-sample evaluation of various values of *k*. Default to 10.
    :type n_folds: number
    
    :param seed: (optional) Random seed used to generate permutations for lrt G0 fitting.
    :type seed: number

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param GB_goal: gigabytes of memory the run should use, optional. If not given, will read the test_snps in blocks the same size as the kernel,
        which is memory efficient with little overhead on computation time.
    :type GB_goal: number

    :param force_full_rank: Even if kernels are defined with fewer SNPs than IIDs, create an explicit iid_count x iid_count kernel. Cannot be True if force_low_rank is True.
    :type force_full_rank: Boolean

    :param force_low_rank: Even if kernels are defined with fewer IIDs than SNPs, create a low-rank iid_count x sid_count kernel. Cannot be True if force_full_rank is True.
    :type force_low_rank: Boolean

    :param mixing: A parameter to LMM learning telling how to combine the two kernels, optional
            If not given will search for best value.
    :type mixing: number

    :param h2: A parameter to LMM learning that tells how much weight to give the K's vs. the identity matrix, optional
            If not given will search for best value.
    :type h2: number

    :param do_plot: If true, will plot, for each chrom, the negative loglikelihood vs k.
    :type do_plot: boolean


    :param runner: a `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`_, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`_

    :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
         alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
    :type count_A1: bool


    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"


    :Example:

    >>> import logging
    >>> import numpy as np
    >>> from fastlmm.association import single_snp_all_plus_select
    >>> from pysnptools.snpreader import Bed
    >>> from fastlmm.util import example_file # Download and return local file name
    >>> from pysnptools.util.mapreduce1.runner import LocalMultiProc
    >>> logging.basicConfig(level=logging.INFO)
    >>> pheno_fn = example_file("fastlmm/feature_selection/examples/toydata.phe")
    >>> test_snps = example_file("fastlmm/feature_selection/examples/toydata.5chrom.*","*.bed")
    >>> snps = Bed(test_snps,count_A1=False)[:,::100] #To make example faster, run on only 1/100th of the data
    >>> chrom5_snps = snps[:,snps.pos[:,0]==5] # Test on only chrom5
    >>> results_dataframe = single_snp_all_plus_select(test_snps=chrom5_snps,G=snps,pheno=pheno_fn,GB_goal=2,runner=LocalMultiProc(20,mkl_num_threads=5), count_A1=False) #Run multiproc
    >>> print(results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe))
    null_9800 0.0793385 4

    """

    #=================================================
    # Start of definition of inner functions
    #=================================================
    def _best_snps_for_each_chrom(chrom_list, input_files, runner, G, n_folds, seed, pheno, covar, force_full_rank, force_low_rank, mixing, h2, k_list, GB_goal):
        #logging.info("Doing GWAS_1K for each chrom and fold. Work_count={0}".format(len(chrom_list)*(n_folds+1)))

        max_k = int(max(k_list))
        assert np.array_equal(G.iid,pheno.iid) and np.array_equal(G.iid,covar.iid), "real assert"

        def mapper_find_best_given_chrom(test_chr):
            G_for_chrom = _K_per_chrom(G, test_chr, G.iid).snpreader
    
            def mapper_gather_lots(i_fold_and_pair):
                i_fold, (train_idx, test_idx) = i_fold_and_pair
                logging.info("Working on GWAS_1K and k search, chrom={0}, i_fold={1}".format(test_chr, i_fold))

                G_train = G_for_chrom[train_idx,:]

                #Precompute whole x whole standardized on train
                from fastlmm.association.single_snp import _internal_determine_block_size, _block_size_from_GB_goal
                min_count = _internal_determine_block_size(G_for_chrom, None, None, force_full_rank, force_low_rank)
                block_size = _block_size_from_GB_goal(GB_goal, G_for_chrom.iid_count, min_count)
                K_whole_unittrain = _SnpWholeWithTrain(whole=G_for_chrom,train_idx=train_idx, standardizer=Unit(), block_size=block_size).read()

                assert np.array_equal(K_whole_unittrain.iid,G_for_chrom.iid),"real assert"
                K_train = K_whole_unittrain[train_idx]
                    
                single_snp_result = single_snp(test_snps=G_train, K0=K_train, pheno=pheno, #iid intersection means when can give the whole covariate and pheno
                             covar=covar, leave_out_one_chrom=False,
                             GB_goal=GB_goal,  force_full_rank=force_full_rank, force_low_rank=force_low_rank, mixing=mixing, h2=h2, count_A1=count_A1)

                is_all = (i_fold == n_folds) if n_folds > 1 else True

                k_list_in =  [0] + [int(k) for k in k_list if 0 < k < len(single_snp_result)]

                if is_all:
                    top_snps = list(single_snp_result.SNP[:max_k])
                else:
                    top_snps = None

                if i_fold == n_folds:
                    k_index_to_nLL = None
                else:
                    k_index_to_nLL = []
                    for k in k_list_in:
                        top_k = G_for_chrom[:,G_for_chrom.sid_to_index(single_snp_result.SNP[:k])]
                        logging.info("Working on chr={0}, i_fold={1}, and K_{2}".format(test_chr,i_fold,k))

                        top_k_train = top_k[train_idx,:] if k > 0 else None
                        fastlmm = FastLMM(force_full_rank=force_full_rank, force_low_rank=force_low_rank,GB_goal=GB_goal)
                        fastlmm.fit(K0_train=K_train, K1_train=top_k_train, X=covar, y=pheno,mixing=mixing,h2raw=h2) #iid intersection means when can give the whole covariate and pheno
    
                        top_k_test = top_k[test_idx,:] if k > 0 else None
                        K0_whole_test = K_whole_unittrain[:,test_idx]
                        nLL = fastlmm.score(K0_whole_test=K0_whole_test,K1_whole_test=top_k_test,X=covar,y=pheno) #iid intersection means when can give the whole covariate and pheno
                        k_index_to_nLL.append(nLL)

                if i_fold > 0:
                    k_list_in = None
    
                return k_list_in, top_snps, k_index_to_nLL

            def reducer_find_best(top_snps_and_k_index_to_nLL_sequence):
                #Starts fold_index+all -> k_index -> nll
                #Need:  k_index -> sum(fold_index -> nll)

                k_index_to_sum_nll = None
                top_snps_all = None
                k_list_in_all = None
                for i_fold, (k_list_in, top_snps, k_index_to_nLL) in enumerate(top_snps_and_k_index_to_nLL_sequence):
                    if k_list_in is not None:
                        assert k_list_in_all is None, "real assert"
                        k_list_in_all = k_list_in
                        k_index_to_sum_nll = np.zeros(len(k_list_in))

                    if top_snps is not None:
                        assert top_snps_all is None, "real assert"
                        top_snps_all = top_snps

                    if k_index_to_nLL is not None:
                        assert i_fold < n_folds or n_folds == 1, "real assert"
                        for k_index, nLL in enumerate(k_index_to_nLL):
                            k_index_to_sum_nll[k_index] += nLL

                #find best # top_snps
                best_k = k_list_in_all[np.argmin(k_index_to_sum_nll)]
                logging.info("For chrom={0}, best_k={1}".format(test_chr,best_k))
                if do_plot: _nll_plot(k_list_in_all, k_index_to_sum_nll)

                #Return the top snps from all
                result = top_snps_all[:best_k]
                return result


            i_fold_index_to_top_snps_and_k_index_to_nLL = map_reduce(
                    _kfold(G_for_chrom.iid_count, n_folds, seed, end_with_all=True),
                    mapper=mapper_gather_lots,
                    reducer=reducer_find_best)
            return i_fold_index_to_top_snps_and_k_index_to_nLL

        chrom_index_to_best_sid = map_reduce(
                chrom_list,
                nested=mapper_find_best_given_chrom,
                input_files=input_files,
                name="best snps for each chrom",
                runner=runner)
        return chrom_index_to_best_sid


    def _gwas_2k_via_loo_chrom(test_snps, chrom_list, input_files, runner, G, chrom_index_to_best_sid, pheno, covar, force_full_rank, force_low_rank, mixing, h2, output_file_name, GB_goal):
        logging.info("Doing GWAS_2K for each chrom. Work_count={0}".format(len(chrom_list)))

        def mapper_single_snp_2K_given_chrom(test_chr):
            logging.info("Working on chr={0}".format(test_chr))
            test_snps_chrom = test_snps[:,test_snps.pos[:,0]==test_chr]
            G_for_chrom = _K_per_chrom(G, test_chr, G.iid).snpreader
            chrom_index = chrom_list.index(test_chr)
            best_sid = chrom_index_to_best_sid[chrom_index]
    
            K1 = G_for_chrom[:,G_for_chrom.sid_to_index(best_sid)]
            result = single_snp(test_snps=test_snps_chrom, K0=G_for_chrom, K1=K1, pheno=pheno,
                        covar=covar, leave_out_one_chrom=False, 
                        GB_goal=GB_goal,  force_full_rank=force_full_rank, force_low_rank=force_low_rank,mixing=mixing,h2=h2,count_A1=count_A1)
            return result
    
        def reducer_closure(frame_sequence): #!!!very similar code in single_snp
            frame = pd.concat(frame_sequence)
            frame.sort_values(by="PValue", inplace=True)
            frame.index = np.arange(len(frame))
            if output_file_name is not None:
                frame.to_csv(output_file_name, sep="\t", index=False)
            logging.info("PhenotypeName\t{0}".format(pheno.sid[0]))
            logging.info("SampleSize\t{0}".format(G.iid_count))
            logging.info("SNPCount\t{0}".format(G.sid_count))
    
            return frame
    
    
        frame = map_reduce(
            chrom_list,
            mapper=mapper_single_snp_2K_given_chrom,
            reducer=reducer_closure,
            input_files=input_files,
            name="single_snp with two K's for all chroms",
            runner=runner
            )
        return frame

    #=================================================
    # End of definition of inner functions
    #=================================================

    #!!!code similar to single_snp
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")
    if k_list is None:
        k_list = np.logspace(start=0, stop=13, num=14, base=2)

    assert test_snps is not None, "test_snps must be given as input"
    test_snps = _snps_fixup(test_snps,count_A1=count_A1)
    G = _snps_fixup(G or test_snps,count_A1=count_A1)
    pheno = _pheno_fixup(pheno,count_A1=count_A1).read()
    assert pheno.sid_count == 1, "Expect pheno to be just one variable"
    pheno = pheno[(pheno.val==pheno.val)[:,0],:]
    covar = _pheno_fixup(covar, iid_if_none=pheno.iid,count_A1=count_A1)
    chrom_list = list(set(test_snps.pos[:,0])) # find the set of all chroms mentioned in test_snps, the main testing data
    G, test_snps, pheno, covar  = pstutil.intersect_apply([G, test_snps, pheno, covar])
    common_input_files = [test_snps, G, pheno, covar]

    chrom_index_to_best_sid = _best_snps_for_each_chrom(chrom_list, common_input_files, runner, G, n_folds, seed, pheno, covar, force_full_rank, force_low_rank, mixing, h2, k_list, GB_goal)

    frame = _gwas_2k_via_loo_chrom(test_snps, chrom_list, common_input_files, runner, G, chrom_index_to_best_sid, pheno, covar, force_full_rank, force_low_rank, mixing, h2, output_file_name, GB_goal)

    return frame

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN) #Needs to be WARN and not INFO to stop Doctest from putting out extra messages.

    import doctest
    doctest.testmod()

