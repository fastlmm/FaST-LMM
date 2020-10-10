#Don't import numpy until after threads are set
import path
import time

cache_top = Path(r'c:\deldir')
output_dir = Path(r'c:\users\carlk\OneDrive\Projects\Science\gpu_pref')

def test_one_exp(test_snps,K0_goal,seed,pheno,covar,leave_out_one_chrom,proc_count,goal_gb):
    K0 = test_snps[:,::test_snps.sid_count//K0_goal]

    runner = None if proc_count == 1 else runner = Local_Multi_Proc(proc_count)

    with patch.dict('os.environ', { 'ARRAY_MODULE': 'cupy' if use_gpu else 'numpy',}
                    ) as patched_environ: #!!!cmk make this a utility

        start_time = time.time()

        results_dataframe = single_snp(K0=K0,test_snps=test_snps, pheno=pheno, covar=covar, 
                                                          leave_out_one_chrom=leave_out_one_chrom, count_A1=False,
                                                          GB_goal=GB_goal, runner=runner)
        delta_time = time.time() - start_time

    perf_result = {'test_snps': str(test_snps),
              'iid_count': test_snps.iid_count,
              'test_sid_count': test_snps.sid_count,
              'K0_sid_count': K0.sid_count,
              'seed': seed,
              'chrom_count': len(np.unique(test_snps.pos[:,0]))
              'covar_count': covar.col_count,
              'leave_out_one_chrom': 1 if leave_out_one_chrom else 0,
              'mkl_thread_count': os.environ['MKL_THREAD_COUNT'],
              'proc_count': proc_count,
              'use_gpu': 1 if use_gpu else 0,
              'time (s)': delta_time,
              }
    return perf_result

def snpsA(seed,iid_count,sid_count):
    chrom_count = 10
    global top_cache
    test_snp_path = cache_top / f'snpsA_{seed}_{chrom_count}_{iid_count}_{sid_count}.bed'
    count_A1 = False
    if not test_snp_path.exists():
        from pysnptools.snpreader import SnpGen
        snpgen = SnpGen(seed=seed,iid_count=iid_count,sid_count=sid_count,chrom_count=chrom_count,block_size=1000)
        test_snps = Bed.write(str(test_snp_path),snpgen.read(dtype='float32'),count_A1=count_A1)
    else:
        test_snps = Bed(str(test_snps_path),count_A1=count_A1)
    from pysnptools.snpreader import SnpData
    np.random.seed(seed)
    pheno = SnpData(iid=test_snps.iid,sid=['pheno'],val=np.random.randn(test_snps.iid_count,1)*3+2)
    covar = SnpData(iid=test_snps.iid,sid=['covar1','covar2'],val=np.random.randn(test_snps.iid_count,2)*2-3)
    return test_snps, pheno, covar

def test_exp_1():
    seed = 1
    iid_count = 1*1000 # number of individuals
    sid_count = 5*1000 # number of SNPs

    mkl_thread_count = 1

    os.environ['MKL_THREAD_COUNT'] = mkl_thread_count
    test_snps, pheno, covar = snpsA(seed,iid_count,sid_count)

    K0_goal = 200
    leave_out_one_chrom = False
    proc_count = 5
    goal_gb = 2

    pref_list = []
    for use_gpu in [False, True]:
        perf_line = test_one_exp(test_snps,K0_goal,seed,pheno,covar,leave_out_one_chrom,proc_count,goal_gb)
        pref_list.append(perf_line)
    pref_df = pd.dataframe(pref_list)
    global output_dir
    pd_write(output_dir /"exp_1_{0}", pref_df)

def pd_write(output_pattern, pref_df)
    i = 0
    while True:
        file = Path(str(output_pattern).format(i))
        if not file.exists():
            break
        i += 1
    pref_df.to_csv(file, sep='\t', index=False)

if __name__ == "__main__":
    test_exp_1()