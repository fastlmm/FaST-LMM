FastLmmSet(
    test="sc_davies",
    outfile = 'tmp/sc_davies_one_kernel_logistic_qqfit.N300.txt',
    phenofile = 'datasets/phenSynthFrom22.23.bin.N300.txt',
    alt_snpreader = 'datasets/all_chr.maf0.001.N300.bed',
    altset_list = 'datasets/set_input.small.txt',
    covarfile  =  None,
    filenull = None,
    autoselect = True,
    mindist = 0,
    idist=2,
    nperm = 0,
    nullfit="qq", #use quantile-quantile fit to estimate params of null distribution
    forcefullrank=False,
    qmax=0.1,      #use the top 10% of null distrib test statistics to fit the null distribution
    write_lrtperm=False,
    datestamp=None,
    nullModel={'link':'logistic', 'effect':'fixed'},
    altModel={'link':'logistic', 'effect':'mixed'},
    log = logging.CRITICAL,
    )
