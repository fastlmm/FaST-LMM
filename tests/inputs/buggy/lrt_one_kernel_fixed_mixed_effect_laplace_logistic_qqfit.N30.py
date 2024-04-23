from fastlmm.pyplink.snpreader.Bed import *


bed = Bed("datasets/all_chr.maf0.001.N30")

# from fastlmm.pyplink.snpreader.Hdf5 import *
# snpMatrix = bed.read()
# Hdf5.write(snpMatrix, 'datasets/all_chr.maf0.001.N30.hdf5')

distributable = FastLmmSet(
    phenofile="datasets/phenSynthFrom22.23.bin.N30.txt",
    alt_snpreader=bed,
    altset_list="datasets/set_input.23.txt",
    covarfile="datasets/all_chr.maf0.001.covariates.N30.txt",
    filenull=None,
    autoselect=False,
    mindist=0,
    idist=2,
    nperm=10,
    test="lrt",
    nullfit="qq",  # use quantile-quantile fit to estimate params of null distribution
    outfile="tmp/lrt_one_kernel_fixed_mixed_effect_laplace_logistic_qqfit.N30.txt",
    forcefullrank=False,
    qmax=0.1,  # use the top 10% of null distrib test statistics to fit the null distribution
    write_lrtperm=True,
    datestamp=None,
    nullModel={"effect": "fixed", "link": "logistic"},
    altModel={
        "effect": "mixed",
        "link": "logistic",
        "approx": "laplace",
        "penalty": None,
    },
    log=logging.CRITICAL,
)
