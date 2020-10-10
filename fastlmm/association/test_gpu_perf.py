#Don't import numpy until after threads are set
import time

def test_one_exp(test_snps,K0_goal,cov,seed,pheno,leave_out_one_chrom,proc_count,goal_gb):
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
              'covar_count': covar.col_count,
              'leave_out_one_chrom': 1 if leave_out_one_chrom else 0,
              'mkl_thread_count': os.environ['MKL_THREAD_COUNT'],
              'proc_count': proc_count,
              'use_gpu': 1 if use_gpu else 0,
              'time (s)': delta_time,
              }
    return perf_result