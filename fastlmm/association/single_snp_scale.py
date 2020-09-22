from __future__ import absolute_import
from __future__ import print_function
import os
import logging
import collections
import numpy as np
import pandas as pd
import scipy.stats as stats
import numpy.linalg as la
import time
from datetime import datetime
import pysnptools.util as pstutil
from pysnptools.snpreader import Bed, Pheno, SnpData, SnpReader, SnpNpz
from pysnptools.kernelreader import KernelData, KernelNpz
from pysnptools.util.mapreduce1 import map_reduce
from pysnptools.util.mapreduce1.mapreduce import _identity, _MapReduce
from pysnptools.util.mapreduce1.runner import Local
from pysnptools.util.intrangeset import IntRangeSet
from pysnptools.util import log_in_place
from pysnptools.util import format_delta
from pysnptools.util import _file_transfer_reporter
from pysnptools.snpreader import SnpMemMap
from pysnptools.standardizer import Unit
from pysnptools.snpreader import SnpGen
from pysnptools.util.filecache import LocalCache, FileCache
from fastlmm.inference.fastlmm_predictor import _snps_fixup, _pheno_fixup
from fastlmm.util.mingrid import minimize1D
from fastlmm.util.matrix.bigsvd import big_sdd
from fastlmm.util.matrix.mmultfile import mmultfile_b_less_aatb,get_num_threads,mmultfile_ata
import six
from six.moves import filter
from six.moves import range

def single_snp_scale(test_snps,pheno,G0=None,covar=None,cache=None,memory_factor=1,
            output_file_name=None, K0=None,
            runner=None,min_work_count=1,
            gtg_runner=None, gtg_min_work_count=None, svd_runner=None, postsvd_runner=None, postsvd_min_work_count=None,test_snps_runner=None, test_snps_min_work_count=None,
            count_A1=False,    
            clear_local_lambda=None, force_python_only=False
            ):
    """
    Function performing single SNP GWAS using REML and cross validation over the chromosomes. Will reorder and intersect IIDs as needed.
    It gives the same results as :func:`.single_snp` but scales a little better on a single machine and has the ability to run on a cluster. (Cluster
    runs require appropriate modules for parameters ``cache`` and ``runner``.)

    Compared to :func:`.single_snp`, :func:`.single_snp_scale` always:

    * does cross validation of chromosomes (:func:`.single_snp`'s ``leave_out_one_chrom=True``)
    * use a low-rank kernel only (:func:`.single_snp`'s ``force_low_rank=True``)
    * uses exactly one kernel constructed from SNPs
    * searches for the best ``h2``. (:func:`.single_snp`'s ``h2=None``)

    :param test_snps: SNPs to test. Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__.
          If you give a string, it should be the name of a `Bed <http://fastlmm.github.io/PySnpTools/#snpreader-bed>`__-formatted file.
          For cluster runs, this and ``G0`` should be readable from any node on the cluster, for example, by using 
          `DistributedBed <http://fastlmm.github.io/PySnpTools/#snpreader-distributedbed>`__ format.
    :type test_snps: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string

    :param pheno: One or more phenotypes: Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__, for example,
           `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__, or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__.
           If you give a string, it should be the name of a `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__-formatted file.
           If only one phenotype is given, any individual with missing phenotype data will be removed from processing.
           If multiple phenotypes are given, any missing data will raise an error. (Avoid this by removing missing values, perhaps via filling in
           missing values.)
    :type pheno: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string

    :param G0: SNPs from which to create a similarity matrix. Defaults to ``test_snps``.
           Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__.
           If you give a string, it should be the name of a `Bed <http://fastlmm.github.io/PySnpTools/#snpreader-bed>`__-formatted file.
           For cluster runs, this and ``test_snps`` should be readable from any node on the cluster, for example, by using 
           `DistributedBed <http://fastlmm.github.io/PySnpTools/#snpreader-distributedbed>`__ format.
    :type G0: `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string

    :param covar: covariate information, optional: Can be any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__, for example, `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__,
           or `SnpData <http://fastlmm.github.io/PySnpTools/#snpreader-snpdata>`__,.
           If you give a string, it should be the name of a `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__-formatted file. #!!!LATER raise error if has NaN
    :type covar: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string

    :param cache: Tells where to store intermediate results. Place the cache on an SSD drive for best performance.
                  By default, the cache will be an automatically-erasing temporary directory. (If the TEMP environment variable is set,
                  Python places the temporary directory under it.)
                  A string can be given and will be interpreted as the path of a local directory to use as a cache. (The local
                  directory will **not** be automatically erased and so must be user managed.) A `FileCache <http://fastlmm.github.io/PySnpTools/#util-filecache-filecache>`__
                  instance can be given, which provides a
                  method to specify a cluster-distributed cache. (`FileCache <http://fastlmm.github.io/PySnpTools/#util-filecache-filecache>`__'s will **not** be automatically erased and must be user managed.)
                  Finally, a dictionary from 0 to 22 (inclusive) to `FileCache <http://fastlmm.github.io/PySnpTools/#util-filecache-filecache>`__ (or None or string)
                  can be given. The dictionary specifies a cache for general work (0) and for every chromosome (1 to 22, inclusive).  
    :type cache: None or string or `FileCache <http://fastlmm.github.io/PySnpTools/#util-filecache-filecache>`__ or dictionary from number to a `FileCache <http://fastlmm.github.io/PySnpTools/#util-filecache-filecache>`__.

    :param memory_factor: How much memory to use proportional to ``G0``, optional.
            If not given, will assume that it can use memory about the same size as one copy of ``G0``.
    :type memory_factor: number

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created. The output format is tab-delimited text.
    :type output_file_name: file name

    :param runner: a `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__

    :param min_work_count: When running the work on a cluster, the **minimum** number of pieces in which to divide the work. Defaults to 1, which is usually fine.
    :type min_work_count: integer

    :param gtg_runner: the `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__ to use instead of ``runner`` for the GtG stage of work.
        For an overview of the stages, see below.
    :type gtg_runner: `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__

    :param gtg_min_work_count: the min_work_count to use instead of ``min_work_count`` on the GtG stage of work.
    :type gtg_min_work_count: integer

    :param svd_runner: the `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__ to use instead of ``runner`` for the SVD stage of work.
    :type svd_runner: `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__

    :param postsvd_runner: the `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__ to use instead of ``runner`` for the PostSVD stage of work.
    :type postsvd_runner: `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__.

    :param postsvd_min_work_count: the min_work_count to use instead of ``min_work_count`` on the PostSVD stage of work.
    :type postsvd_min_work_count: integer

    :param test_snps_runner: the `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__ to use instead of ``runner`` for the TestSNPS stage of work.
    :type test_snps_runner: a `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__.

    :param test_snps_min_work_count: the min_work_count to use instead of ``min_work_count`` on the TestSNPS stage of work.
    :type test_snps_min_work_count: integer

    :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
         alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
    :type count_A1: bool

    :param clear_local_lambda: A function to run in the middle of the PostSVD stage, typically to clear unneeded large files from the local file cache.
    :type clear_local_lambda: function or lambda

    :param force_python_only: (Default: False) Skip faster C++ code. Used for debugging and testing.

    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"

    :Example:

    >>> import logging
    >>> from fastlmm.association import single_snp
    >>> from pysnptools.snpreader import Bed
    >>> from fastlmm.util import example_file # Download and return local file name
    >>> logging.basicConfig(level=logging.INFO)
    >>> test_snps = Bed(example_file('tests/datasets/synth/all.*','*.bed'),count_A1=True)[:,::10] #use every 10th SNP
    >>> pheno_fn = example_file("tests/datasets/synth/pheno_10_causals.txt")
    >>> cov_fn = example_file("tests/datasets/synth/cov.txt")
    >>> results_dataframe = single_snp_scale(test_snps=test_snps, pheno=pheno_fn, covar=cov_fn, count_A1=False)
    -etc-
    >>> print(results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe))
    snp1200_m0_.37m1_.36 0.0 500

    The stages of processing are:

    * 0: G - Read G0 (the selected SNPs), standardize, regress out covariates, output G
    * 1: GtG - Compute GtG = G.T x G
    * 2: SVD - For each chromosome in the TestSNPs, extract the GtG for the chromosome, compute the singular value decomposition (SVD) on that GtG
    * 3: PostSVD - Transform the 22 GtG SVD into 22 G SVD results called U1 .. U22. Find h2, the importance of person-to-person similarity.
    * 4: TestSNPs - For each test SNP, read its data, regress out covariates, use the appropriate U and compute a Pvalue.

    All stages cache intermediate results. If the results for stage are found in the cache, that stage will be skipped.
    """

    #Fill in with the default runner and default min_work_count
    gtg_runner = gtg_runner or runner
    gtg_min_work_count = gtg_min_work_count or min_work_count
    svd_runner = svd_runner or runner
    postsvd_runner = postsvd_runner or runner
    postsvd_min_work_count = postsvd_min_work_count or min_work_count
    test_snps_runner = test_snps_runner or runner
    test_snps_min_work_count = test_snps_min_work_count or min_work_count

    #######################################
    # all in time and space 1M x 3
    #######################################
    G0 = G0 or K0 or test_snps
    chrom_list, pheno1, RxY, test_snps1, X, Xdagger, G0 = preload(covar, G0, pheno, test_snps, count_A1=count_A1, multi_pheno_is_ok=True)
    cache_dict = _cache_dict_fixup(cache,chrom_list)

    G0_memmap_lambda, ss_per_snp = get_G0_memmap(G0, cache_dict[0], X, Xdagger, memory_factor)

    gtg_npz_lambda = get_gtg(cache_dict[0], G0.iid_count, G0.sid, G0_memmap_lambda, memory_factor, gtg_runner, min_work_count=gtg_min_work_count)

    svd(chrom_list, gtg_npz_lambda, memory_factor, cache_dict[0], G0.iid_count, G0.pos, ss_per_snp, X, svd_runner)

    log_frequency = 200 if logging.getLogger().level <= logging.INFO else 0
    postsvd(chrom_list, gtg_npz_lambda, memory_factor, cache_dict, G0.iid, G0.sid, G0_memmap_lambda, ss_per_snp, RxY, X, postsvd_runner, clear_local_lambda, postsvd_min_work_count,log_frequency=log_frequency)

    test_snps_memory_factor = memory_factor

    frame = do_test_snps(cache_dict, chrom_list, gtg_npz_lambda, test_snps_memory_factor, G0.iid_count, G0.sid_count, G0.pos, pheno1,
                         ss_per_snp, RxY, X, Xdagger, test_snps=test_snps1, runner=test_snps_runner,
                         output_file_name=output_file_name, min_work_count=test_snps_min_work_count)

    return frame




def snp_fn(data_folder,file_index):
    return data_folder + "/{0}.bed".format(file_index)
def pheno_fn(data_folder,file_index):
    return data_folder + "/pheno.{0}.txt".format(file_index)
def covar_fn(data_folder,file_index):
    return data_folder + "/covar.{0}.txt".format(file_index)

def generate(data_folder,iid_count,file_count,sid_count,seed=0):
    from pysnptools.util import snp_gen
    from pysnptools.util.generate import _generate_phenotype

    for file_index in range(file_count):
        snp_filename = snp_fn(data_folder,file_index)
        pheno_filename = pheno_fn(data_folder,file_index)
        covar_filename = covar_fn(data_folder,file_index)

        if file_index > 0:
            iid = [[iid[0],"{0}.{1}".format(iid[1],file_index)] for iid in full_snp_data.iid]
            if not os.path.exists(snp_filename):
                snp_data1 = SnpData(iid=iid,sid=full_snp_data.sid,val=full_snp_data.val,pos=full_snp_data.pos)
                Bed.write(snp_filename,snp_data1)
            if not os.path.exists(pheno_filename):
                pheno_data1 = SnpData(iid=iid,sid=phenodata.sid,val=phenodata.val)
                Pheno.write(pheno_filename,pheno_data1)
            if not os.path.exists(covar_filename):
                covar_data1 = SnpData(iid=iid,sid=covardata.sid,val=covardata.val)
                Pheno.write(covar_filename,covar_data1)

        else:
            #########################
            # SNPs
            #########################
            if not os.path.exists(snp_filename):
                logging.info("Creating '{0}'".format(snp_filename))
                #for 1000 x 4M takes about 6 clock minutes and 36 gig of memory
                full_snp_data = snp_gen(fst=.1, dfr=0, iid_count=iid_count // file_count, sid_count=sid_count, chr_count=22, label_with_pop=True,seed=seed)
                # writing 1000 x 4M takes about 1.5 clock minutes (1 proc), 1 gig on 
                Bed.write(snp_filename,full_snp_data)
                some_snp_data = full_snp_data[:,::100]
            else:
                full_snp_data = Bed(snp_filename).read()
                some_snp_data = Bed(snp_filename)[:,::100].read() #Reading takes 5 clock secs (1 proc) and .5 gig of memory

            #########################
            # Pheno
            #########################
            if not os.path.exists(pheno_filename):
                logging.info("Creating '{0}'".format(pheno_filename))
                phenodata = SnpData(iid=some_snp_data.iid,sid=["pheno"],val=_generate_phenotype(some_snp_data, 10, genetic_var=.5, noise_var=.5, seed=seed).reshape(-1,1))
                Pheno.write(pheno_filename,phenodata)
            else:
                phenodata = Pheno(pheno_filename).read()

            #########################
            # Covar
            #########################
            if not os.path.exists(covar_filename):
                logging.info("Creating '{0}'".format(covar_filename))
                np.random.seed(seed)
                cov_val_0 = phenodata.val[:,0] * .0001 + np.random.normal(size=some_snp_data.iid_count)*1
                cov_val_1 = np.random.normal(size=some_snp_data.iid_count)*2
                covardata = SnpData(iid=some_snp_data.iid,sid=["covar0", "covar2"],val=np.c_[cov_val_0,cov_val_1])
                Pheno.write(covar_filename,covardata)
            else:
                covardata = Pheno(covar_filename).read()

def read_pheno_cache(data_folder,file_index_end,prefix,fn_fp):
    merge_fn = data_folder + "/{0}.0.{1}.merge.npz".format(prefix,file_index_end)
    if not os.path.exists(merge_fn):
        snpreader = _MergeIIDs([Pheno(fn_fp(data_folder,file_index)) for file_index in range(file_index_end)]).read()
        SnpNpz.write(merge_fn,snpreader)
    else:
        snpreader = SnpNpz(merge_fn)
    return snpreader

def bed_list(data_folder,snp0,file_index_end,iid_list_or_none=None):
    return [Bed(snp_fn(data_folder,file_index),
                sid=snp0.sid,pos=snp0.pos,
                iid = (None if iid_list_or_none is None else iid_list_or_none[file_index]),
                skip_format_check=True) for file_index in range(file_index_end)]
def read_bed_cache(data_folder, file_index_end):
    snp0 = Bed(snp_fn(data_folder,0),skip_format_check=True)
    snps_iid_list_fn = data_folder + "/snps.0.{0}.iid_lists.npz".format(file_index_end)
    if os.path.exists(snps_iid_list_fn):
        with np.load(snps_iid_list_fn) as data:
            iid_list = data['iid_list']
    else:
        iid_list = [bed.iid for bed in bed_list(data_folder,snp0,file_index_end,None)]
        np.savez(snps_iid_list_fn,iid_list=iid_list)
    snps_merge_fn = data_folder + "/snps.0.{0}.mergeiids.npz".format(file_index_end)
    snp = _MergeIIDs(bed_list(data_folder,snp0,file_index_end,iid_list),cache_file=snps_merge_fn,skip_check=True)
    return snp

#def scale_test():
#    import numpy as np
#    import numpy.linalg as la
#    import time
#    for sid_count in xrange(10000,50000,2500):
#        #iid_count = sid_count * 1000000 / 50000
#        iid_count = sid_count
#        x = np.random.randint(3, size=(sid_count,iid_count)).T
#        #print "creating ata"
#        #t = time.time()
#        #ata = np.dot(x.T,x)
#        #diff0 = time.time()-t
#        #print "ata ready"
#        t = time.time()
#        [a,b,c] = la.svd(x,False,True) #!!!use big_svd?
#        diff = time.time()-t
#        print iid_count,sid_count,diff#0,diff

#!!! exactly the same as in single_snp.py
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

def compute_stats(start, stop, snpsKY, snpsKsnps, YKY, N, logdetK,  iid_count, sid, pos, h2, covar_bias_count, pheno_or_none):
    '''
    Computes the stats for the test SNPs. Contains no slow matrix work. All work is order #test SNPs
    '''
    #==============================================================
    # 4M -> 4M
    #    (all the final operations)
    #==============================================================
    beta = snpsKY / snpsKsnps
    if np.isnan(beta.min()):
        logging.warning("NaN beta value seen, may be due to an SNC (a constant SNP)")
        beta[snpsKY==0] = 0.0
    
    variance_explained_beta = (snpsKY * beta)
    r2 = YKY[np.newaxis,:] - variance_explained_beta 
    variance_beta = r2 / (N - 1) / snpsKsnps
    fraction_variance_explained_beta = variance_explained_beta / YKY[np.newaxis,:] # variance explained by beta over total variance
    sigma2 = r2 / N
    nLL = 0.5 * (logdetK + N * (np.log(2.0 * np.pi * sigma2) + 1))
    chi2stats = beta*beta/variance_beta
    p_values = stats.f.sf(chi2stats,1,iid_count-(covar_bias_count+1))[:,0]#note that G.shape is the number of individuals
    
    dataframe = _create_dataframe(len(sid))
    if pheno_or_none is not None:
        dataframe['Pheno'] = pheno_or_none
    dataframe['sid_index'] = np.arange(start,stop)
    dataframe['SNP'] = np.array(sid,'str') #This will be ascii on Python2 and unicode on Python3
    dataframe['Chr'] = pos[:,0]
    dataframe['GenDist'] = pos[:,1]
    dataframe['ChrPos'] = pos[:,2]
    dataframe['PValue'] = p_values
    dataframe['SnpWeight'] = beta[:,0]
    dataframe['SnpWeightSE'] = np.sqrt(variance_beta[:,0])
    dataframe['SnpFractVarExpl'] = np.sqrt(fraction_variance_explained_beta[:,0])
    dataframe['Mixing'] = np.zeros((len(sid))) + 0
    dataframe['Nullh2'] = np.zeros((len(sid))) + h2
    return dataframe

def _internal_single2(G0_chrom, test_snps_chrom, pheno, covar):
    return frame

def preload(covar, G0, pheno, test_snps, count_A1, multi_pheno_is_ok=False):
    """
    Load 'covar' and 'pheno' into memory. Remove and reorder iids as needed from all inputs.
    Compute some other values based on 'covar' and 'pheno'
    """
    test_snps = _snps_fixup(test_snps, count_A1=count_A1)
    G0 = _snps_fixup(G0, count_A1=count_A1)
    pheno = _pheno_fixup(pheno,count_A1=count_A1).read()

    if not multi_pheno_is_ok:
        assert pheno.sid_count == 1, "Expect pheno to be just one variable"

    if pheno.sid_count == 1:
        pheno = pheno[(pheno.val==pheno.val)[:,0],:]
    else:
        assert not np.isnan(pheno.val).any(), "When multiple phenotypes are given, they must not have any missing values."

    covar = _pheno_fixup(covar, iid_if_none=pheno.iid,count_A1=count_A1)
    chrom_list = list(set(test_snps.pos[:,0])) # find the set of all chroms mentioned in test_snps, the main testing data
    G0, test_snps, pheno, covar  = pstutil.intersect_apply([G0, test_snps, pheno, covar]) #!!!is this time well spent?
    
    X = np.c_[covar.read(view_ok=True,order='A').val,np.ones((covar.iid_count, 1))]  #view_ok because np.c_ will allocation new memory
    y =  pheno.read(view_ok=True,order='A').val #view_ok because this code already did a fresh read to look for any missing values 
    
    #=============================================================
    # 1M x 3 -> 3 x 1M
    #    Only need to do this once, not 22 times
    #=============================================================
    Xdagger = la.pinv(X)       #SVD-based, and seems fast
    #=============================================================
    # 3 x 1M x 2 => 3 x 2
    #  Move out of chrom loop
    #=============================================================
    beta_y = Xdagger.dot(y)
    #=============================================================
    # 1M x 3 x 2 => 1M x 2
    #  Move out of chrom loop
    #=============================================================
    RxY = y - X.dot(beta_y)
    return chrom_list, pheno, RxY, test_snps, X, Xdagger, G0

def get_clear_local(file_cache):

    def sub_dir_path(d):
        return list(filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)]))

    def clear_local_lambda():
        for local_root in set(storage.local_lambda()[1] for storage in six.itervalues(file_cache)):
            for sub in sub_dir_path(local_root):
                logging.info("Removing '{0}'".format(sub))
                shutil.rmtree(sub)

    return clear_local_lambda
                
def get_G0_memmap(G0, file_cache_parent, X, Xdagger, memory_factor):
    """
    Create a version of the selected SNPs that is unit standardized. It will also be 'DiagKToN' standardization ready (see PySnpTools for info on 'DiagKToN'). The result
    will be uploaded into a file_cache (e.g. some shared cluster storage). The actual return value from this function is a lambda that when called will download G0 (if needed) and open it
    up as a memory-mapped numpy array.
    """
    file_cache = file_cache_parent.join('0_G')
    fn_done = "done.txt"
    fn_G0_memmap = "G0_data.memmap"
    fn_ss = "ss_per_snp.npz"

    def G0_memmap_lambda():
        with file_cache.open_read(fn_G0_memmap) as local_G0_memmap:
            G0_memmap = SnpMemMap(local_G0_memmap)
        #!!!not fully closing the read here because SnpMemMap has the local file open, but at least we know usage has started
        return G0_memmap

    if file_cache.file_exists(fn_done):
        with file_cache.open_read(fn_ss) as ss_storage:
            with np.load(ss_storage) as data:
                ss_per_snp = data['arr_0']
        return G0_memmap_lambda, ss_per_snp

    file_cache.rmtree('')

    assert file_cache.file_exists(fn_G0_memmap) == file_cache.file_exists(fn_ss), "expect '{0}' and '{1}' to either both exist or neither".format(fn_G0_memmap, fn_ss)


    work_count2 = -(G0.iid_count // -int(G0.sid_count*memory_factor)) #Find the work count based on batch size (rounding up)
    work_count2 = max(work_count2,2) #Just to test it
    def debatch_closure2(work_index2):
        return G0.sid_count * work_index2 // work_count2

    logging.info("About to allocate memmap of G0_data.memmap")

    with _file_transfer_reporter("G0_data.memmap upload", size=0, updater=None) as updater2:
        with file_cache.open_write(fn_G0_memmap,size=8*G0.iid_count*G0.sid_count,updater=updater2) as G0_memmap_storage_file_name:
            G0_data_memmap = SnpMemMap.empty(iid=G0.iid,sid=G0.sid,filename=G0_memmap_storage_file_name,pos=G0.pos,order='F') #!!!is this the best order? dtype default ok?
            logging.info("Finished with allocation of memmap of G0_data.memmap")
            ss_per_snp = np.empty([G0.sid_count])
            t0 = time.time()
            for work_index in range(work_count2):
                if work_count2 > 1: logging.info("get_G0_data_memmap: Working on part {0} of {1}".format(work_index,work_count2))
                start = debatch_closure2(work_index)
                stop = debatch_closure2(work_index+1)
    
                ##############################################################
                ############# BIG ############################################
                ##############################################################
                # 1M x 25K -> 1M x 25K
                # Read all SNPs and Unit standardize
                #    Can be clustered by SNP, although output is 200 GB
                #            (Just remember unit trained and apply later?
                #   Use memory mapped files? SPARK? would be nice if could run on carlk4
                #===========================================================
                G0_data_piece = G0[:,start:stop].read().standardize(Unit())
                ss_per_snp[start:stop] = np.einsum('i...,i...',G0_data_piece.val,G0_data_piece.val) #Find the (G0_data.val**2).sum(0) without memory allocation
                #!!!Might be slightly better if this code and the original FaSTLMM code standardized the diagonal AFTER regressing away the covariants
    
                ##=============================================================
                ## 3 x 1M x 25K -> 3 x 25K
                ##=============================================================
                beta_G = Xdagger.dot(G0_data_piece.val)
    
                ##=============================================================
                # 1M x 3 x 25K -> 3 x 25K
                #   should be able to do this in place
                #=============================================================
                G0_data_piece.val-=X.dot(beta_G)#mem x 2
                G0_data_memmap.val[:,start:stop] = G0_data_piece.val
            t1=time.time()    
            logging.info("G0 work took {0}".format(format_delta(t1-t0)))
            logging.info("About to get name of file to np.savez ss_per_snp")
            with file_cache.open_write(fn_ss) as ss_storage:
                logging.info("About to np.savez ss_per_snp in '{0}'".format(ss_storage))
                np.savez(ss_storage, ss_per_snp)
                logging.info("About to close ss_per_snp")
            t2=time.time()
            logging.info("done with ss_per_snp in {0}".format(format_delta(t2-t1)))

            logging.info("About to SnpMemMap.close G0_data_memmap in '{0}'".format(G0_memmap_storage_file_name))

            G0_data_memmap.flush()
            t3=time.time()
            logging.info("done with SnpMemMap.close G0_data_memmap in {0}".format(format_delta(t3-t2)))
            logging.info("About to close g0.npz and g0.memmap")

    file_cache.save(fn_done,'')
    t4=time.time()
    logging.info("Done with g0.npz and g0.memmap in {0}".format(format_delta(t4-t3)))

    return G0_memmap_lambda, ss_per_snp

def get_gtg(common_cache_parent, G0_iid_count, G0_sid, G0_memmap_lambda, memory_factor,  runner, min_work_count=1,force_python_only=False):
    """
    Compute G0.T x G0. Do the calculation in pieces with the smallest atom of work being done in multithreaded C++ code streaming the input matrix values from an file on (SSD) disk.
    The product, GtG will be stored in some cluster storage location.
    The actual return value of this function is a lambda. When that lambda is called, it retrieves GtG from storage and returns a local copy.
    """
    common_cache = common_cache_parent.join('1_GtG')
    fn_done = "done.txt"
    fn_gtg = "gtg.npz"

    def reader_closure():
        with common_cache.open_read(fn_gtg) as local_fn_gtg:
            gtg_npz = KernelNpz(local_fn_gtg)
        return gtg_npz

    if common_cache.file_exists(fn_done):
        return reader_closure

    common_cache.rmtree('')
    G0_sid_count = len(G0_sid)

    def writer_closure(gtg_data):
        with common_cache.open_write(fn_gtg, gtg_data.iid_count**2*8) as local_gtg:
            gtg_npz = KernelNpz.write(local_gtg,gtg_data)
        return gtg_npz

    ##############################################################
    ############# SLOWEST#########################################
    ##############################################################
    # 25K x 1M x 25K -> 25K x 25K
    #=============================================================

    logging.info("Computing G0.T x G0")
    work_count = -(G0_iid_count // -int(G0_sid_count*memory_factor)) #Find the work count based on batch size (rounding up)
    work_count = max(min_work_count,work_count) # Always divide the work into at least min_work_count parts
    work_count = min(work_count,len(G0_sid)) #Don't divide the work more than the number of test SNPs

    t0_gtg = time.time()
    name="{0}.get_gtg".format(os.path.basename(common_cache.name)) #!!! little bug: if tree_cache.name ends with "/" or "\" doesn't work
    mmultfile_ata(G0_memmap_lambda,writer_closure,G0_sid,work_count,name,runner=runner,force_python_only=force_python_only)
    common_cache.save(fn_done,'')
    logging.info("gtg: clocktime {0}".format(format_delta(time.time()-t0_gtg)))

    return reader_closure


def get_U(chrom, chrom_cache):
    chrom = int(chrom)

    fn_U = "3_PostSVD/chrom{0}/U.memmap".format(chrom) #!!!const
    fn_UUYetc = "3_PostSVD/chrom{0}/UUYetc.npz".format(chrom) #!!!const

    assert chrom_cache.file_exists(fn_U) == chrom_cache.file_exists(fn_UUYetc), "expect '{0}' and '{1}' to either both exist or neither".format(fn_U, fn_UUYetc)
    assert chrom_cache.file_exists(fn_U), "expect '{0}'".format(fn_U)

    with chrom_cache.open_read(fn_UUYetc) as local_fn_UUYetc:
        with np.load(local_fn_UUYetc) as data:
            S    = data['S']
            UY   = data['UY']
            UUY  = data['UUY']

    with chrom_cache.open_read(fn_U) as local_fn_U:
        U_memmap = SnpMemMap(local_fn_U)

    return S, U_memmap, UY, UUY

def apply_h2(h2,S,UYUY,UUYUUYsum0,N,k): #!!!make more matrix like? (or multiproc???)
    result_list = []
    for pheno_index in range(len(UUYUUYsum0)):
        result = apply_h2_inner(h2[pheno_index],S,UYUY[:,[pheno_index]],UUYUUYsum0[[pheno_index]],N,k)
        result_list.append(result)
    logdetK = np.array([logdetK for logdetK, YKY, Sd, denom in result_list])
    YKY = np.array([YKY for logdetK, YKY, Sd, denom in result_list]).reshape(-1)
    Sd = np.array([Sd for logdetK, YKY, Sd, denom in result_list]).transpose()
    denom = np.array([denom for logdetK, YKY, Sd, denom in result_list])
    return logdetK, YKY, Sd, denom

def apply_h2_inner(h2,S,UYUY,UUYUUYsum0,N,k):
    Sd = (h2 * S + (1.0 - h2)) #25K
    denom = 1.0 - h2
    YKY = (UYUY / Sd.reshape(-1,1)).sum(0) + UUYUUYsum0 / denom #Can be done in blocks
    logdetK = np.log(Sd).sum() + (N - k) * np.log(denom)
    return logdetK, YKY, Sd, denom

def find_h2(k, N, UUYUUYsum0, UYUY, S, chrom):
    h2_list = []
    for pheno_index in range(len(UUYUUYsum0)):
        h2_inner = find_h2_inner(k, N, UUYUUYsum0[[pheno_index]],UYUY[:,[pheno_index]],S,chrom)
        h2_list.append(h2_inner)
    return np.array(h2_list)

def find_h2_inner(k, N, UUYUUYsum0, UYUY, S, chrom):

    resmin = [None]
    def f(x,resmin=resmin):
        h2 = float(x)
        logdetK, YKY, _, _ = apply_h2_inner(h2,S,UYUY,UUYUUYsum0,N,k)
        sigma2 = YKY / N
        nLL = 0.5 * (logdetK + N * (np.log(2.0 * np.pi * sigma2) + 1))
        if (resmin[0] is None) or (nLL < resmin[0]['nLL']):
            resmin[0] = {'nLL':nLL,'h2':h2}
        logging.debug("search\t{0}\t{1}".format(h2,nLL))
        return nLL
    
    #=============================================================
    # 25K x about 30
    #=============================================================
    _ = minimize1D(f=f, nGrid=10, minval=0, maxval=.99999)
    h2 = resmin[0]['h2']
    logging.info("h2={0}".format(h2))

    logdetK, YKY, Sd, denom = apply_h2_inner(h2,S,UYUY,UUYUUYsum0,N,k)
    return h2

def get_h2(k, N, UUYUUYsum0, UYUY, S, chrom_cache, chrom):
    fn_h2 = "3_PostSVD/chrom{0}/h2.npz".format(int(chrom)) #!!!const
    with chrom_cache.open_read(fn_h2) as local_fn_h2:
        with np.load(local_fn_h2) as data:
            h2    = data['arr_0']
    logdetK, YKY, Sd, denom = apply_h2(h2,S,UYUY,UUYUUYsum0,N,k)
    return h2, logdetK, YKY, Sd, denom

def svd(chrom_list, gtg_npz_lambda, memory_factor, common_cache_parent, G0_iid_count, G0_pos, ss_per_snp, X, runner_svd):
    """
    For the chromosomes listed, compute an SVD on a square matrix SNP-to-SNP matrix. Each SVD can be done on a different
    node in a cluster. The actual SVD is done with special version of the LAPACK DGESDD function.

    Results are stored at a known location in the cluster storage.
    """

    common_cache = common_cache_parent.join('2_SVD')

    needed_chrom_list = [chrom for chrom in chrom_list if not common_cache.file_exists('done{0}.txt'.format(int(chrom)))]
    if not needed_chrom_list:
        return []

    def mapper_closure(chrom):
        logging.info("Caching chrom {0}".format(chrom))
        gtg_npz = gtg_npz_lambda()
        idx = G0_pos[:,0] != chrom
        factor = float(G0_iid_count)/ss_per_snp[idx].sum()

        ##############################################################
        ############# SLOWEST ########################################
        ##############################################################
        # 1M x 25K x 25K => 1M x 25K
        #=============================================================
        idx2 = np.arange(len(idx))[idx]
        ata = gtg_npz[idx2].read()
        

        #Because the iid_count can be so big, the factor created for one_step_svd can be very inappropriate for here
        factor_tall_skinny = float(ata.iid_count) / np.diag(ata.val).sum()
        ata.val *= factor_tall_skinny
    
        ##############################################################
        ############# SLOWEST ########################################
        ##############################################################
        # 25K x 25K x 25K -> 25K x 25K
        #=============================================================
        num_threads = get_num_threads()
        logging.info("About to svd on square {0}. Expected time ({2} procs)={1}".format(ata.iid_count,format_delta((ata.iid_count*.000707)**3*20.0/num_threads),num_threads))
        t0 = time.time()
        [Uata3,Sata3,_] = big_sdd(ata.val) #wrecks ata.val
        logging.info("Actual time for svd on square={0}".format(format_delta(time.time()-t0)))
        Sata3 *= (factor / factor_tall_skinny) #make the results match one_step_svd
        S3 = Sata3**.5
        V3 = Uata3.T
    
        ##############################################################
        ############# SLOWEST ########################################
        ##############################################################
        # 25K**2.8 -> 25K x 25K
        #        Is there a faster way to do the dot with the diag???
        #=============================================================
        SVinv3 = la.inv(np.dot(np.diag(S3), V3))
        SVinv3 *= np.sqrt(factor)
    
        S = S3
        if np.any(S < -0.1):
            logging.warning("kernel contains a negative Eigenvalue")
        inonzero = S > 1E-10
        S = S[inonzero]
        S = S * S

        SVinv3b=SVinv3[:,inonzero]

        fn = "SVinv_etc{0}.npz".format(int(chrom)) #!!!const
        if common_cache.file_exists(fn):
            common_cache.remove(fn)
        with common_cache.open_write(fn) as local_file_name:
            np.savez(local_file_name, SVinv3b=SVinv3b,S=S)
        common_cache.save('done{0}.txt'.format(int(chrom)),'')
        return fn

    SVinv_etc_fn_list = map_reduce(needed_chrom_list,
                        mapper=mapper_closure,
                        runner=runner_svd,
                        name="{0}.svd".format(os.path.basename(common_cache.name)),
                        input_files=[],
                        output_files=[]
                        )
    return SVinv_etc_fn_list

def postsvd_piece(start_iid_index, stop_iid_index, G0_memmap, idx_array, SVinv3b, fn_U_piece, log_frequency=-1):
    logging.info("About to read")
    t0_piece = time.time()
    piece = np.zeros((stop_iid_index-start_iid_index,SVinv3b.shape[0]),order='F')
    with open(G0_memmap.filename,"rb") as fp:
        so_far_idx = 0
        for start_idx,stop_idx in idx_array:
            start_idx,stop_idx = int(start_idx),int(stop_idx) #We convert from numpy.int32 to python ints so that their product will be a python long instead of a (overflowed) numpy.int32
            fp.seek(G0_memmap.offset+start_idx*G0_memmap.iid_count*8) #Skip sids that are not of interest
            for sid_index in range(so_far_idx,so_far_idx+stop_idx-start_idx):
                if log_frequency > 0 and sid_index % log_frequency == 0:
                    logging.info("on sid_index {0} of {1}".format(sid_index,SVinv3b.shape[0]))
                row = np.fromfile(fp, dtype=np.float64, count=G0_memmap.iid_count) #!!! instead of reading whole column, how about just reading the piece of the column that is wanted? Whould this be a little faster?
                try:
                    piece[:,sid_index] = row[start_iid_index:stop_iid_index]
                except:
                    raise Exception("piece[:,{0}] = row[{1}:{2}]".format(sid_index,start_iid_index,stop_iid_index))
            so_far_idx += stop_idx-start_idx
    logging.info("About to mult")
    product = np.array(np.dot(piece,SVinv3b),order='F')
    logging.info("About to save")   
    product.flatten(order='K').tofile(fn_U_piece,'')
    logging.info("post svd piece: clocktime {0}".format(format_delta(time.time()-t0_piece)))

def postsvd(chrom_list, gtg_npz_lambda, memory_factor, cache_dict, G0_iid, G0_sid, G0_memmap_lambda, ss_per_snp, RxY, X, postsvd_runner, clear_local_lambda, min_work_count, log_frequency=-1):
    """
    For the chromosomes listed, take the square SVD and turn it into U, the "tall-and-skinny" SVD needed. Also, computed related values UY and UUY.
    Finally, find search for the best h2, which tells how much weight to give to person-to-person similarity vs. pure noise.

    This function uses two levels of map-reduce that are run as a single cluster job. The top level loops across the chromosomes, the
    second level does a matrix multiple in blocks. At the lowest level, the matrix multiple is multithreaded.

    Save the results under a known name in the cluster storage.
    """
    sub_dir = '3_PostSVD'
    common_cache = cache_dict[0].join(sub_dir)
    #cache_dict[int(chrom)].file_exists("h2_{0}.npz".format(int(chrom)))
    needed_chrom_list = [chrom for chrom in chrom_list if not common_cache.file_exists("chrom{0}/done.txt".format(int(chrom)))]
    if not needed_chrom_list:
        return
    G0_iid_count = len(G0_iid)
    G0_sid_count = len(G0_sid)

    def mapper_closure(chrom):
        chrom = int(chrom)
        chrom_storage = cache_dict[chrom].join(sub_dir)

        fn_U = "chrom{0}/U.memmap".format(int(chrom))
        if chrom_storage.file_exists(fn_U):
            chrom_storage.remove(fn_U)
        fn_UUYetc = "chrom{0}/UUYetc.npz".format(int(chrom))
        if chrom_storage.file_exists(fn_UUYetc):
            chrom_storage.remove(fn_UUYetc)
        fn_h2 = "chrom{0}/h2.npz".format(int(chrom)) #!!!const
        if chrom_storage.file_exists(fn_h2):
            chrom_storage.remove(fn_h2)

        work_count = -(G0_iid_count // -int(G0_sid_count*memory_factor)) #Find the work count based on batch size (rounding up)
        work_count = max(work_count,min_work_count)

        def mapper_closure_inner(work_index):

            SVinv_etc_fn = "2_SVD/SVinv_etc{0}.npz".format(chrom)
            with cache_dict[0].open_read(SVinv_etc_fn) as handle_local:
                with np.load(handle_local) as data:
                    SVinv3b   = data['SVinv3b']
                    S    = data['S']

            G0_memmap = G0_memmap_lambda()
            idx = G0_memmap.pos[:,0] != chrom
            ##############################################################
            ############# SLOWEST ########################################
            ##############################################################
            # 1M x 25K x 25K => 1M x 25K

            sid = ["sid{0}".format(i) for i in range(SVinv3b.shape[1])]
            idx_intrangeset = IntRangeSet(i for i,keep in enumerate(idx) if keep) #IntRangeSet('285:1000')
            idx_array = np.array(list(idx_intrangeset.ranges()))


            def debatch_closure(work_index):
                return G0_memmap.iid_count * work_index // work_count

            if work_count > 1: logging.info("post svd, chrom {0}: Working on part {1} of {2}".format(chrom, work_index, work_count))

            start_iid_index = debatch_closure(work_index)
            stop_iid_index = debatch_closure(work_index+1)

            fn_U_done = "chrom{0}/U_pieces/done{1}_{2}.txt".format(chrom,work_index,work_count)
            fn_U_piece = "chrom{0}/U_pieces/{1}_{2}.memmap".format(chrom,work_index,work_count)
            chrom_cache = cache_dict[chrom].join(sub_dir)
            if chrom_cache.file_exists(fn_U_done):
                logging.info("Piece '{0}' already exists, so skipping work".format(fn_U_piece))
            else:
                if chrom_cache.file_exists(fn_U_piece):
                    chrom_cache.remove(fn_U_piece)
                with chrom_cache.open_write(fn_U_piece) as local_file_name:
                    postsvd_piece(start_iid_index, stop_iid_index, G0_memmap, idx_array, SVinv3b, local_file_name, log_frequency=log_frequency)
                chrom_cache.save(fn_U_done,'')
            return fn_U_piece, start_iid_index, stop_iid_index, len(sid) if work_index==0 else None, S if work_index==0 else None

        def reducer_closure_inner(fn_U_piece_sequence):
            if clear_local_lambda is not None:
                logging.info("To save space, removing all cached files from local drive")
                clear_local_lambda()
                
            t0_start = time.time()
            logging.info("Starting postsvd reduce for chrom '{0}' with file downloads".format(chrom))

            fn_U_piece_list = list(fn_U_piece_sequence) #make it a list instead of generator, so can run through it twice
            _,_,_, sid_count, S = fn_U_piece_list[0] #Get sid_count and S from the first work item. We pass these this way to avoid bring in files to the local machine (if run with "map_reduceX") or to the reduce 

            for fn_U_piece, start_iid_index, stop_iid_index, _, _ in fn_U_piece_list:
                with chrom_storage.open_read(fn_U_piece) as local_file_name: #!!! is assuming that a local file with the name is OK.
                    pass
                
            t0_download = time.time()
            logging.info("File downloads took {0}. Next putting U together".format(format_delta(t0_download-t0_start)))

            with chrom_storage.open_write(fn_U) as handle_fn_U_file_name:
                sid = ["sid{0}".format(i) for i in range(sid_count)]    
                U_snp_mem_map = SnpMemMap.empty(iid=G0_iid,sid=sid,filename=handle_fn_U_file_name,dtype=np.float64, order='F')
                fp_list = [] #!!! would be nice to have a try_catch to be sure all these get closed
                file_name_list = []
                for piece_index, (fn_U_piece, start_iid_index, stop_iid_index, _, _) in enumerate(fn_U_piece_list):
                    with chrom_storage.open_read(fn_U_piece) as handle_local: #The local file is used after this is closed. Not nice, but OK in this case.
                        logging.info("About to open piece {0} of {1}".format(piece_index,len(fn_U_piece_list)))
                        fp_list.append(open(handle_local,"rb"))
                with open(U_snp_mem_map.filename,"r+b") as fp_U:
                    fp_U.seek(U_snp_mem_map.offset)
                    with log_in_place("Creating U file", logging.INFO) as log_writer:
                        for sid_index in range(sid_count):
                            if sid_index % 100 == 0: #!!!use something like log_freq instead of '100'
                                log_writer("On sid {0} of {1}".format(sid_index,sid_count))
                            prev_stop = 0
                            for piece_index, (fn_U_piece, start_iid_index, stop_iid_index, _, _) in enumerate(fn_U_piece_list):
                                assert start_iid_index == prev_stop, "real assert"
                                prev_stop = stop_iid_index
                                fp = fp_list[piece_index]
                                buf2 = np.fromfile(fp, dtype=np.float64, count=stop_iid_index-start_iid_index)
                                buf2.tofile(fp_U)
                            assert stop_iid_index == G0_iid_count, "real assert"
                for fp in fp_list:
                    fp.close()

                t0_U = time.time()
                logging.info("Putting U together took {0}. Next creating UY & UUY files".format(format_delta(t0_U - t0_download)))
            
                #=============================================================
                # 25K x 1M x 2 -> 25K x 2
                #=============================================================
                UY = U_snp_mem_map.val.T.dot(RxY)  #Note: This could be pushed into the 'map' step, with each step returning a UYi that would all be summed together.
                #=============================================================
                # 1M x 25K x 2 -> 1M x 2
                #=============================================================
                UUY = RxY - U_snp_mem_map.val.dot(UY) #This is can't be pushed into the 'map' step because it depends on all of UY

                U_snp_mem_map.flush()


            with chrom_storage.open_write(fn_UUYetc) as local_fn_UUYetc:
                np.savez(local_fn_UUYetc, S=S, UY=UY, UUY=UUY)


            #=============================================================
            # 25K x about 30
            #=============================================================
            UYUY = UY * UY
            UUYUUYsum0 = (UUY * UUY).sum(0)
            N = G0_iid_count - X.shape[1] #number of degrees of freedom
            k = S.shape[0]
            h2 = find_h2(k, N, UUYUUYsum0, UYUY, S, chrom) #h2 depends on y

            t0_etc = time.time()
            logging.info("Creating related files took {0}. Next uploading files.".format(format_delta(t0_etc-t0_U)))

            with chrom_storage.open_write(fn_h2) as local_fn_h2:
                np.savez(local_fn_h2, h2)

            if clear_local_lambda is not None:
                chrom_storage.cloud_storage_only()

            common_cache.save("chrom{0}/done.txt".format(int(chrom)),'')

            t0_up = time.time()
            logging.info("Uploading took {0}.".format(format_delta(t0_up - t0_etc)))

            logging.info("Postsvd Reduce Summary: Down {0}, Calc {1}, Up {2}, Total {3}".format(
                format_delta(t0_download-t0_start),
                format_delta(t0_etc - t0_download),
                format_delta(t0_up - t0_etc),
                format_delta(t0_up - t0_start)
                ))

        return map_reduce(range(work_count),
                            mapper=mapper_closure_inner,
                            reducer=reducer_closure_inner,
                            name="{0}.postsvd_{1}".format(os.path.basename(chrom_storage.name),chrom),
                            input_files=[],
                            output_files=[],
                        )

    map_reduceX(needed_chrom_list,
                        mapper=mapper_closure,
                        name="{0}.postsvd".format(os.path.basename(cache_dict[0].name)),
                        runner=postsvd_runner
                    )

def get_U_h2(chrom, gtg_npz_lambda, memory_factor, chrom_cache, G0_iid_count, G0_pos, ss_per_snp, RxY, pheno, X):
    #!!! similar code above
    logging.debug("Retrieving on chrom {0}".format(chrom))
    idx = G0_pos[:,0] != chrom
    factor = float(G0_iid_count)/ss_per_snp[idx].sum()

    ##############################################################
    ############# SLOWEST ########################################
    ##############################################################
    # 25K x 25K x 25K -> 25K x 25K
    # 1M x 25K x 25K => 1M x 25K
    #=============================================================
    S, U_memmap, UY, UUY = get_U(chrom, chrom_cache) #!!!y: at the end, U is multiplied twice with yish things
    
    #=============================================================
    # 25K x about 30
    #=============================================================
    UYUY = UY * UY
    UUYUUYsum0 = (UUY * UUY).sum(0)
    N = pheno.iid_count - X.shape[1] #number of degrees of freedom
    k = S.shape[0]
    h2, logdetK, YKY, Sd, denom = get_h2(k, N, UUYUUYsum0, UYUY, S, chrom_cache, chrom) #!!!y: h2 depends on y
                
    return h2, U_memmap, Sd, denom, UY, UUY, YKY, N, logdetK

def __del__(test_snps):
    '''
    Close any open Bed files
    '''
    from pysnptools.snpreader._subset import _SnpSubset
    from pysnptools.snpreader import SnpData, _Distributed1Bed, DistributedBed
    from pysnptools.pstreader import _MergeRows, _MergeCols

    if isinstance(test_snps, Bed) or isinstance(test_snps, _Distributed1Bed):
        test_snps.__del__()
    elif isinstance(test_snps, SnpGen) or isinstance(test_snps, SnpData):
        pass
    elif isinstance(test_snps, DistributedBed):
        __del__(test_snps._merge) 
    elif isinstance(test_snps, _SnpSubset):
        __del__(test_snps._internal) 
    elif isinstance(test_snps, _MergeRows) or isinstance(test_snps, _MergeCols):
        for sub in test_snps.reader_list:
            __del__(sub)
    else:
        logging.warning("Don't know how to __del__ '{0}'".format(test_snps))

# code: no longer used (was for debugging???)
#def is_good(node,file,chrom):
#    #Only use the file if it 1. exists and 2. It's *.memmap is over 150000000000L in bytes in size
#    if not os.path.exists(file):
#        return False
#    big = pattern2.format(node,".U{0}.0.memmap".format(int(chrom)))
#    if not os.path.exists(big) or os.path.getsize(big) <  150000000000L:
#        return False
#    mid = pattern2.format(node,".U{0}.0.npz".format(int(chrom)))
#    if not os.path.exists(mid) or os.path.getsize(mid) <  5000000L:
#        return False
#    return True

#def random_source(chrom,pattern,node_list):
#    part = pattern.format(int(chrom))
#    shuffled_node_list = list(node_list[int(chrom)-1])
#    random.shuffle(shuffled_node_list)
#    for node in shuffled_node_list:
#        source = pattern2.format(node,part)
#        if is_good(node,source,chrom):
#            return source
#    raise Exception("Can't find '{0}' in {1}".format(part, node_list))

def do_test_snps(cache_dict, chrom_list, gtg_npz_lambda, memory_factor, G0_iid_count, G0_sid_count, G0_pos, pheno, ss_per_snp, RxY, X, Xdagger, test_snps, runner, output_file_name=None, min_work_count=1):
    """
    For every test SNP, measure its pvalue. Nest two levels of map-reduce. On the top level, loop over chromosomes. Within each chromosome
    loop over blocks of testSNPs. At the lowest-level, do the matrix-multiple (and related operators) with multithreaded C++.

    Return a Pandas frame of the final results.
    """
    __del__(test_snps) #Close any open Bed files

    results_storage = cache_dict[0].join('4_TestSNPs')

    def mapper_closure2(chrom):
        chrom_index = chrom_list.index(chrom)
        chrom = int(chrom)
        chrom_storage = cache_dict[chrom]

        test_snps_chrom = test_snps[:,test_snps.pos[:,0]==chrom]
        logging.info("test_snps has sid_count of {0}".format(test_snps.sid_count))
        logging.info("test_snps_chrom({1}) has sid_count of {0}".format(test_snps_chrom.sid_count,chrom))

        work_count = int(-(-test_snps_chrom.iid_count // (G0_sid_count*memory_factor))) # Do blocks based on the size of G0
        work_count = max(min_work_count,work_count) # Always divide the work into at least min_work_count parts
        work_count = min(work_count,test_snps_chrom.sid_count) #Don't divide the work more than the number of test SNPs
        logging.info("work_count = {0}".format(work_count))

        def mapper_closure(work_index):

            done_file = 'chrom{0}/done.{1}of{2}.txt'.format(chrom,work_index,work_count)
            cache_file = 'chrom{0}/result.{1}of{2}.tsv'.format(chrom,work_index,work_count)
            if results_storage.file_exists(done_file):
                with results_storage.open_read(cache_file) as local_file:
                    dataframe = pd.read_csv(local_file,delimiter = '\t')
                    return dataframe

            if results_storage.file_exists(cache_file):
                results_storage.remove(cache_file)

            def debatch_closure(work_index):
                start = test_snps_chrom.sid_count * work_index // work_count
                logging.debug("chrom={0},work_index={1},start={2},test_snps_chrom.sid_count={3}".format(chrom,work_index,start,test_snps_chrom.sid_count))
                return start
    
            h2, U_memmap, Sd, denom, UY, UUY, YKY, N, logdetK = get_U_h2(chrom, gtg_npz_lambda, memory_factor, chrom_storage, G0_iid_count, G0_pos, ss_per_snp, RxY, pheno, X)

            if work_count > 1: logging.info("single_low_snp: Working on snp block {0} of {1}".format(work_index,work_count))
            start = debatch_closure(work_index)
            logging.debug("A:chrom={0},work_index={1},start={2}".format(chrom,work_index,start))
            stop = debatch_closure(work_index+1)
            logging.debug("A:chrom={0},work_index+1={1},end={2}".format(chrom,work_index+1,stop))
            t0_gen = time.time()
            snps_read = test_snps_chrom[:,start:stop].read().standardize()
            logging.info("test snp reader {0} of {1}, clocktime {2}".format(work_index,work_count,format_delta(time.time()-t0_gen)))
            logging.debug("Xdagger {0}x{1}. snp_read {2}x{3}".format(Xdagger.shape[0],Xdagger.shape[1],snps_read.iid_count,snps_read.sid_count))
    
            #=============================================================
            #  Note: on all these the 4M is all the SNPs, but because we can cluster by chrom and blocks really 4M/22/10 (about 16K to 18K) per work item
            # 3 x 1M x 4M -> 3 x 4M
            #=============================================================
            logging.debug("beta_snps = Xdagger.dot(snps_read.val)")
            beta_snps = Xdagger.dot(snps_read.val)
            ##############################################################
            ########### BIG ##############################################
            ##############################################################
            # 1M x 3 x 4M -> 1M x 4M
            #=============================================================
            logging.debug("Rxsnps = snps_read.val - X.dot(beta_snps)")
            Rxsnps = snps_read.val - X.dot(beta_snps)
            ##############################################################
            ###########   SLOWEST ########################################
            ##############################################################
            # 25K X 1M x 4M -> 25K x 4M
            #=============================================================
            logging.debug("U_memmap '{0}'".format(U_memmap))
            logging.debug("U_data {0}x{1}. Rxsnps {2}x{3}. time={4}".format(U_memmap.iid_count,U_memmap.sid_count,Rxsnps.shape[0],Rxsnps.shape[1],datetime.now().strftime("%Y-%m-%d %H:%M")))
            logging.debug("Usnps = U_data.val.T.dot(Rxsnps)")

            ##############################################################
            ###########   SLOWEST ##### BIG ##############################
            ##############################################################
            # 25K x 1M x 4M -> 25K x 4M
            # and 
            # 25K x 4M 
            #          while at it, also (UUsnps * UUsnps).sum(0)
            #=============================================================
            #U_data = U_memmap.read(order='K',view_ok=True) #in the work loop so it can be run on cluster
            #UUsnps = Rxsnps - U_data.val.dot(Usnps)

            Usnps, UUsnps = mmultfile_b_less_aatb(U_memmap, Rxsnps, log_frequency=10 if logging.getLogger().level <= logging.INFO else -1)
            logging.debug("U_data {0}x{1}. Usnps {2}x{3}. time={4}".format(U_memmap.iid_count,U_memmap.sid_count,Usnps.shape[0],Usnps.shape[1],datetime.now().strftime("%Y-%m-%d %H:%M")))
            logging.debug("UUsnps = Rxsnps - U_data.val.dot(Usnps)")
    
            dataframe_list = []
            UUSnpsUUSnps_sum0 = (UUsnps * UUsnps).sum(0) 
            for pheno_index in range(len(h2)): #!!!make more matrix like? (or multiprocess???)
                #==============================================================
                # 25K x 4M -> 4M
                #==============================================================
                logging.debug("time={0}".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
                logging.debug("snpsKsnps = ((Usnps / Sd.reshape(-1,1) * Usnps).sum(0) + (UUsnps * UUsnps).sum(0) / denom).reshape(-1,1) #Can be done in blocks")
                snpsKsnps = ((Usnps / Sd[:,[pheno_index]] * Usnps).sum(0) + UUSnpsUUSnps_sum0 / denom[pheno_index]).reshape(-1,1) #Can be done in blocks
                #==============================================================
                # 25K x 4M -> 25K x 4M
                #==============================================================
                logging.debug("UAS = Usnps / np.lib.stride_tricks.as_strided(Sd, (Sd.size,Usnps.shape[1]), (Sd.itemsize,0))")
                UAS = Usnps / np.lib.stride_tricks.as_strided(Sd[:,[pheno_index]], (Sd.shape[0],Usnps.shape[1]), (Sd.itemsize,0))
                #==============================================================
                # 4M x 25K x 1 + 4M x 1M x 1 => 4M
                #==============================================================
                logging.debug("snpsKY = UAS.T.dot(UY) + UUsnps.T.dot(UUY) / denom #!!!y: some y work")
                snpsKY = UAS.T.dot(UY[:,[pheno_index]]) + UUsnps.T.dot(UUY[:,[pheno_index]]) / denom[pheno_index]
                #==============================================================
                # 4M
                #==============================================================
                assert test_snps_chrom.iid_count == U_memmap.iid_count
                pheno_or_none = None if pheno.sid_count == 1 else pheno.sid[pheno_index]
                dataframe_inner = compute_stats(start, stop, snpsKY, snpsKsnps, YKY[[pheno_index]], N, logdetK[[pheno_index]], snps_read.iid_count, snps_read.sid, 
                                          snps_read.pos, h2[pheno_index], covar_bias_count=Xdagger.shape[0], pheno_or_none=pheno_or_none)
                dataframe_list.append(dataframe_inner)

            dataframe = pd.concat(dataframe_list)
            with results_storage.open_write(cache_file) as local_file:
                dataframe.to_csv(local_file,sep = '\t',index=False)

            results_storage.save(done_file,'')

            return dataframe
    
        def reducer_closure(result_list):
            frame = pd.concat(result_list)
            frame.sort_values(by="PValue", inplace=True)
            frame.index = np.arange(len(frame))
            return frame
    
        frame = map_reduce(range(work_count),
                   mapper=mapper_closure,
                   name="{0}.test_snps_{1}".format(os.path.basename(cache_dict[0].name),chrom),
                   reducer=reducer_closure,
                   )
        return frame
    
    def reducer_closure2(result_list2):
        frame = pd.concat(result_list2)
        frame.sort_values(by="PValue", inplace=True)
        frame.index = np.arange(len(frame))

        if output_file_name is not None:
            pstutil.create_directory_if_necessary(output_file_name)
            frame.to_csv(output_file_name, sep="\t", index=False)

        logging.info("PhenotypeName(s)\t{0}".format(pheno.sid))
        logging.info("SampleSize\t{0}".format(test_snps.iid_count))
        logging.info("SNPCount\t{0}".format(test_snps.sid_count))
    
        return frame
    
    name_prefix = "{0}.test_snps_".format(os.path.basename(cache_dict[0].name))
    frame = map_reduceX(chrom_list,
                        mapper=mapper_closure2,
                        reducer=reducer_closure2,
                        name=lambda index:name_prefix+str(int(chrom_list[index])), #Create a job name based on the chrom number
                        runner=runner
                )

    return frame

def clear_cache_dict(start_stage,cache_dict,chrom_num_list):
    """
    Remove selected intermediate results from the cache dictionary.

    :param start_stage: None (do nothing), 'all' (clear all), 'gtg' (clear stage gtg and later),
         'svd' (clear stage 'svd' and later), 'postsvd' (clear stage postsvd and later), 'testsnps' (clear stage testsnps)
    :type start_stage: string or None

    :param cache_dict: Dictionary mapping chromosomes to :class:`LocalCache` or other storage
    :type cache_dict: dictionary

    :param chrom_num_list: List of chromosome numbers to look at.
    :type chrom_num_list: list of numbers.

    """
    if start_stage is None:
        return

    with log_in_place("clearing cache dict to '{0}'".format(start_stage), logging.INFO) as log_writer:
        _clear_cache_dict_internal(start_stage,cache_dict,chrom_num_list,log_writer)

def _clear_cache_dict_internal(start_stage,cache_dict,chrom_num_list,log_writer):
    if start_stage == "all":
        _clear_cache_dict_internal("gtg",cache_dict,chrom_num_list,log_writer) #Recursion
        for file_name in ["G0_data.memmap","ss_per_snp.npz"]:
            if cache_dict[0].file_exists(file_name):
                cache_dict[0].remove(file_name,log_writer=log_writer)
    elif start_stage == "gtg":
        _clear_cache_dict_internal("svd",cache_dict,chrom_num_list,log_writer) #Recursion
        for file_name in ["gtg.npz"]:
            log_writer(file_name)
            if cache_dict[0].file_exists(file_name):
                cache_dict[0].remove(file_name,log_writer=log_writer)
    elif start_stage == "svd":
        _clear_cache_dict_internal("postsvd",cache_dict,chrom_num_list,log_writer) #Recursion
        for chrom in chrom_num_list:
            log_writer("svd chrom {0}".format(chrom))
            cache_dict[0].remove_from_cloud_storage("SVinv_etc{0}.npz".format(chrom))
    elif start_stage == "postsvd":
        #Doing the postsvd deletes before the testsnps deletes is faster because it removes whole containers in a single operation.
        for chrom in chrom_num_list:
            log_writer("postsvd: chrom {0}".format(chrom))
            cache_dict[chrom].remove_from_cloud_storage(log_writer=log_writer)
        _clear_cache_dict_internal("testsnps",cache_dict,chrom_num_list,log_writer) #Recursion
    elif start_stage == "testsnps":
        for file_name in ["G0_data.memmap","ss_per_snp.npz"]:
            log_writer("cloud storage only '{0}'".format(file_name))
            if cache_dict[0].file_exists(file_name):
                cache_dict[0].cloud_storage_only(file_name,log_writer=log_writer)
        for chrom in [0]+chrom_num_list:
            log_writer("cloud storage only '{0}'".format(chrom))
            cache_dict[chrom].cloud_storage_only(log_writer=log_writer)
    else:
        raise Exception("Don't know start_stage='{0}'".format(start_stage))


def _cache_dict_fixup(cache_dict,chrom_list):
    #If a dictionary, then fix up the values. Else, fix up the value and create a dictionary.
    if isinstance(cache_dict, collections.Mapping):
        return {k:FileCache._fixup(v,default_subfolder='single_snp_scale') for k,v in six.iteritems(cache_dict)}
    else:
        cache_value = FileCache._fixup(cache_dict,default_subfolder='single_snp_scale')
        return {chrom:cache_value for chrom in [0]+chrom_list}


#!!!if is useful, move near pysnptools's mapreduce.py (with better name)
def map_reduceX(input_seq, mapper=_identity, reducer=list, runner=None,name=None):
    '''
    Function for running a function on sequence of inputs and running a second function on the results. Can be nested and clusterized.
    For each top-level input, a separate job will be created.


    :param input_seq: a sequence of inputs. The sequence must support the len function and be indexable. e.g. a list, xrange(100)
    :type input_seq: a sequence

    :param mapper: A function to apply to each set of inputs (optional). Defaults to the identity function. (Also see 'mapper')
    :type mapper: a function

    :param reducer: A function to turn the results from the mapper to a single value (optional). Defaults to creating a list of the results.
    :type reducer: a function that takes a sequence

    :param name: A name to be displayed if this work is done on a cluster.
    :type name: a string

    :param runner: a runner, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: a runner.

    :rtype: The results from the reducer.

    '''
    if runner is None:
        runner = Local()

    if name==None:
        name = str(distributable_list[0]) or ""
        if len(distributable_list) > 1:
            name += "-etc"

    with _MapReduce._dyn_vars(is_in_nested=True):
        distributable_list = [mapper(input) for input in input_seq]
    if hasattr(runner,"run_list"):
        return runner.run_list(distributable_list,reducer=reducer,name=name)
    else:
        result_list = [runner.run(distributable) for distributable in distributable_list]
        top_level_result = reducer(result_list)
        return top_level_result


if __name__ == "__main__":
    import doctest
    import logging
    #logging.getLogger().setLevel(logging.WARN)

    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod(optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE)
    doctest.ELLIPSIS_MARKER = '...'
    print('done')
