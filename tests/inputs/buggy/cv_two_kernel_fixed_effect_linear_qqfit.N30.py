import logging
from fastlmm.association.FastLmmSet import FastLmmSet

distributable = FastLmmSet(
    phenofile = 'datasets/phenSynthFrom22.23.bin.N30.txt',
    bedfilealt = 'datasets/all_chr.maf0.001.N30',
    altset_list = 'datasets/set_input.23.txt',
    covarfile  =  None,
    filenull = 'datasets/all_chr.maf0.001.chr22.23.N30.bed',
    autoselect = True,
    mindist = 0,
    idist=2,
    nperm = 3,
    test="cv",
    outfile = 'tmp/cv_two_kernel_fixed_effect_linear_qqfit.N30.txt',
    forcefullrank=False,
    qmax=0.1,
    write_lrtperm=False,
    datestamp=None,
    nullModel = {'effect':'fixed',
                 'link':'logistic',
                 'penalty':'l1'},
    altModel = {'effect':'fixed',
                 'link':'logistic',
                 'penalty':'l1'},
    log = logging.CRITICAL,
    )
