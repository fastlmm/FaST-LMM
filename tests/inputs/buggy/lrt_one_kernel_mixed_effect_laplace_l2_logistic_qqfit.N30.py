distributable = FastLmmSet(
    phenofile = 'datasets/phenSynthFrom22.23.bin.N30.txt',
    alt_snpreader = 'datasets/all_chr.maf0.001.N30',
    altset_list = 'datasets/set_input.23_17_11.txt',
    covarfile  =  None,
    filenull = None,
    autoselect = False,
    mindist = 0,
    idist=2,
    nperm = 10,
    test="lrt",
    nullfit="qq", #use quantile-quantile fit to estimate params of null distribution
    outfile = 'tmp/lrt_one_kernel_mixed_effect_laplace_l2_logistic_qqfit.N30.txt',
    forcefullrank=False,
    qmax=0.1,      #use the top 10% of null distrib test statistics to fit the null distribution
    write_lrtperm=True,
    datestamp=None,
    nullModel={'effect':'mixed', 'link':'logistic',
               'approx':'laplace', 'penalty':'l2'},
    altModel={'effect':'mixed', 'link':'logistic',
              'approx':'laplace', 'penalty':'l2'},
    log = logging.CRITICAL,
    detailed_table = False
    )
