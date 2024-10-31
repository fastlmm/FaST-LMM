import logging
from fastlmm.association.FastLmmSet import FastLmmSet

distributable = FastLmmSet(
    test="sc_davies",
    outfile = 'tmp/sc_davies_two_kernel_qqfit.N30.txt',
    phenofile = 'datasets/phenSynthFrom22.23.N30.txt',
    bedfilealt = 'datasets/all_chr.maf0.001.N30',
    altset_list = 'datasets/set_input.small.txt',
    covarfile  =  None,
    filenull = 'datasets/all_chr.maf0.001.chr22.23.N30.bed',
    autoselect = True,
    mindist = 0,
    idist=2,
    nperm = 0,
    nullfit="qq", #use quantile-quantile fit to estimate params of null distribution
    forcefullrank=False,
    qmax=0.1,      #use the top 10% of null distrib test statistics to fit the null distribution
    write_lrtperm=False,
    datestamp=None,
    nullModel={'link':'logistic'},
    altModel=None,
    log = logging.CRITICAL,
    )
