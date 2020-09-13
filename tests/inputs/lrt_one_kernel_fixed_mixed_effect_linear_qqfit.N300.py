FastLmmSet(
    phenofile = 'datasets/phenSynthFrom22.23.N300.txt',
    alt_snpreader = 'datasets/all_chr.maf0.001.N300.bed',
    altset_list = 'datasets/set_input.23.txt',
    covarfile  =  None,
    filenull = None,
    autoselect = False,
    mindist = 0,
    idist=2,    
    nperm = 10,
    test="lrt",
    nullfit="qq", #use quantile-quantile fit to estimate params of null distribution
    outfile = 'tmp/lrt_one_kernel_fixed_mixed_effect_linear_qqfit.N300.txt',
    forcefullrank=False,
    qmax=0.1,      #use the top 10% of null distrib test statistics to fit the null distribution
    write_lrtperm=True,
    datestamp=None,
    nullModel={'effect':'fixed', 'link':'linear'},
    altModel={'effect':'mixed', 'link':'linear'},
    log = logging.CRITICAL,
    )
