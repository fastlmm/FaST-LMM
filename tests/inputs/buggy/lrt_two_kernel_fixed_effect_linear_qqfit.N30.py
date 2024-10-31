import logging
from fastlmm.association.FastLmmSet import FastLmmSet

distributable = FastLmmSet(
    phenofile = 'datasets/phenSynthFrom22.23.N30.txt',
    bedfilealt = 'datasets/all_chr.maf0.001.N30',
    altset_list = 'datasets/set_input.23.txt',
    covarfile  =  None,
    filenull = 'datasets/all_chr.maf0.001.chr22.23.N30.bed',
    autoselect = False,
    mindist = 0,
    idist=2,
    nperm = 10,
    test="lrt",
    nullfit="qq", #use quantile-quantile fit to estimate params of null distribution
    outfile = 'tmp/lrt_two_kernel_fixed_effect_linear_qqfit.N30.txt',
    forcefullrank=False,
    qmax=0.1,      #use the top 10% of null distrib test statistics to fit the null distribution
    write_lrtperm=True,
    datestamp=None,
    nullModel={'effect':'mixed', 'link':'linear'},
    altModel={'effect':'mixed', 'link':'linear'},
    log = logging.CRITICAL,
    )
