from __future__ import absolute_import
from __future__ import print_function
from pysnptools.util.mapreduce1.runner import *
import logging
import fastlmm.pyplink.plink as plink
import numpy as np
from fastlmm.inference.lmm_cov import LMM as fastLMM
import scipy.stats as stats
from fastlmm.util.pickle_io import load, save
import time
import pandas as pd

def snp_set(
        test_snps,
        set_list,
        pheno,
        covar = None,
        output_file_name = None,
        G0 = None,
        test="lrt",
        write_lrtperm = False,
        nperm = 10,
        npermabs = None,
        mpheno=1,
        G0_fit="qq",
        qmax=0.1,
        seed =   None,
        minsetsize = None,
        maxsetsize = None,
        mindist=0,
        idist=1,
        show_pvalue_5050 = False
    ):
    """
    Function performing GWAS on sets of snps


    :param test_snps: The base name of the file containing the SNPs for alternative kernel. The file must be in PLINK Bed format.
    :type test_snps: a string

    :param set_list: The name of a tab-delimited file defining the sets. The file should contain two-columns 'snp' and 'set'.
    :type set_list: a string

    :param pheno: The name of a file containing the phenotype. The file must be in PLINK phenotype format.
    :type pheno: a string

    :param covar: covariate information, optional: The name of a file in PLINK phenotype format.
    :type covar: a 'pheno dictionary' or a string

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param G0: Training SNPs from which to construct a similarity kernel. It should be the base name of files in PLINK Bed or Ped format.
    :type G0: a string

    :param test: 'lrt' (default) or 'sc_davies'
    :type test: a string

    :param write_lrtperm: (default: False) If True, write the lrtperm vector (dictated by seed) to a file.
    :type write_lrtperm: boolean

    :param nperm: (default: 10) number of permutations per test
    :type nperm: number

    :param npermabs: (default: None) absolute number of permutations
    :type npermabs: number

    :param mpheno: (default: 1) integer, starting at 1, representing the index of the phenotype tested
    :type mpheno: number

    :param G0_fit: (default: "qq") How to fit G0. Should be either "qq" or "ml"
    :type G0_fit: string

    :param qmax: (default: .1) Use the top qmax fraction of G0 distrib test statistics to fit the G0 distribution
    :type qmax: number

    :param seed: (optional) Random seed used to generate permutations for lrt G0 fitting.
    :type seed: number

    :param minsetsize: (optional) only include sets at least this large (inclusive)
    :type minsetsize: number

    :param maxsetsize: (optional) only include sets no more than this large (inclusive)
    :type maxsetsize: number

    :param mindist: (default 0) SNPs within mindist from the test SNPs will be removed from
    :type mindist: number

    :param idist: (default: 1) the type of position to use with mindist
         1, genomic distance
         2, base-pair distance
    :type idist: number

    :param show_pvalue_5050: (default: False) show a conservative P-value arising from an assumed null distribution that is a 50-50 mixture distribution of 0 and 1 dof chi squares [Molenberghs and Verbeke, 2003].
     Provided for backwards compatibility.

    :type show_pvalue_5050: Boolean


    :rtype: Pandas dataframe with one row per set.

    :Example:

    >>> import logging
    >>> from fastlmm.association import snp_set
    >>> logging.basicConfig(level=logging.INFO)
    >>> result_dataframe = snp_set(
    ...     test_snps = '../../tests/datasets/all_chr.maf0.001.N300',
    ...     set_list = '../../tests/datasets/set_input.23.txt',
    ...     pheno = '../../tests/datasets/phenSynthFrom22.23.N300.txt')
    >>> print(result_dataframe.iloc[0].SetId, round(result_dataframe.iloc[0]['P-value'],15))
    set23 0.0

    """

    assert test=="lrt" or test=="sc_davies", "Expect test to be 'lrt' or 'sc_davies'"

    if G0 is None:
        nullModel={'effect':'fixed', 'link':'linear'}
        altModel={'effect':'mixed', 'link':'linear'}
    else:
        nullModel={'effect':'mixed', 'link':'linear'}
        altModel={'effect':'mixed', 'link':'linear'}
        if test=="lrt":
            test="lrt_up"


    if output_file_name is None:
        import tempfile
        fileno, output_file_name = tempfile.mkstemp()
        fptr= os.fdopen(fileno)
        is_temp = True
    else:
        is_temp = False


    from fastlmm.association.FastLmmSet import FastLmmSet
    fast_lmm_set = FastLmmSet(
        outfile=output_file_name,
        phenofile=pheno,
        alt_snpreader=test_snps,
        altset_list=set_list,
        covarfile=covar,
        filenull=G0,
        nperm=nperm,
        mindist=mindist,
        idist=idist,
        mpheno=mpheno,
        nullfit = G0_fit,
        qmax=qmax,
        test=test,
        autoselect=False,
        nullModel=nullModel,
        altModel=altModel,
        npermabs = npermabs,
        calseed = seed,
        minsetsize = minsetsize,
        maxsetsize = maxsetsize,
        write_lrtperm = write_lrtperm,
        show_pvalue_5050 = show_pvalue_5050,
        )
    result = Local().run(fast_lmm_set)

    dataframe=pd.read_csv(output_file_name,delimiter='\t',comment=None) #Need \t instead of \s because the output has tabs by design and spaces in column names(?)
    if is_temp:
        fptr.close()
        os.remove(output_file_name)

    return dataframe
    

if __name__ == "__main__":

    import doctest
    doctest.testmod()

    print("done")

