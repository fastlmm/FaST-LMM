FastLmmSet(   # noqa: F821
    test="sc_davies",
    outfile = 'tmp/sc_davies_two_kernel_linear_qqfit.N300.needsReorder.txt',
    phenofile = 'datasets/phenSynthFrom22.23.N300.randcidorder.txt',
    alt_snpreader = 'datasets/all_chr.maf0.001.N300',
    altset_list = 'datasets/set_input.small.txt',
    covarfile  =  None,
    filenull = 'datasets/all_chr.maf0.001.chr22.23.N300.bed',
    autoselect = True,
    mindist = 0,
    idist=2,
    nperm = 0,
    nullfit="qq", #use quantile-quantile fit to estimate params of null distribution
    forcefullrank=False,
    qmax=0.1,      #use the top 10% of null distrib test statistics to fit the null distribution
    write_lrtperm=False,
    datestamp=None,
    nullModel={'link':'linear', 'effect':'mixed'},
    altModel={'link':'linear', 'effect':'mixed'},
    log = logging.CRITICAL,  # noqa: F821
    )
