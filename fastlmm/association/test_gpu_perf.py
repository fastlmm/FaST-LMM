#Don't import numpy until after threads are set
from pathlib import Path
import time
import os
import logging

if False:
    cache_top = Path(r'c:\deldir')
    output_dir = Path(r'c:\users\carlk\OneDrive\Projects\Science\gpu_pref')
else:
    cache_top = Path(r'm:\deldir')
    output_dir = Path(r'd:\OneDrive\Projects\Science\gpu_pref')


def test_one_exp(test_snps,K0_goal,seed,pheno,covar,leave_out_one_chrom,use_gpu,proc_count,GB_goal):
    import numpy as np
    from pysnptools.util.mapreduce1.runner import LocalMultiProc
    from unittest.mock import patch
    from fastlmm.association import single_snp

    K0 = test_snps[:,::test_snps.sid_count//K0_goal]

    runner = None if proc_count == 1 else LocalMultiProc(proc_count)

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
              'chrom_count': len(np.unique(test_snps.pos[:,0])),
              'covar_count': covar.col_count,
              'leave_out_one_chrom': 1 if leave_out_one_chrom else 0,
              'mkl_thread_count': os.environ['MKL_THREAD_COUNT'],
              'use_gpu': 1 if use_gpu else 0,
              'proc_count': proc_count,
              'GB_goal': GB_goal,
              'time (s)': delta_time,
              }
    return perf_result

def snpsA(seed,iid_count,sid_count):
    import numpy as np
    from pysnptools.snpreader import Bed
    from pysnptools.snpreader import SnpGen

    chrom_count = 10
    global top_cache
    test_snp_path = cache_top / f'snpsA_{seed}_{chrom_count}_{iid_count}_{sid_count}.bed'
    count_A1 = False
    if not test_snp_path.exists():
        snpgen = SnpGen(seed=seed,iid_count=iid_count,sid_count=sid_count,chrom_count=chrom_count,block_size=1000)
        test_snps = Bed.write(str(test_snp_path),snpgen.read(dtype='float32'),count_A1=count_A1)
    else:
        test_snps = Bed(str(test_snp_path),count_A1=count_A1)
    from pysnptools.snpreader import SnpData
    np.random.seed(seed)
    pheno = SnpData(iid=test_snps.iid,sid=['pheno'],val=np.random.randn(test_snps.iid_count,1)*3+2)
    covar = SnpData(iid=test_snps.iid,sid=['covar1','covar2'],val=np.random.randn(test_snps.iid_count,2)*2-3)
    return test_snps, pheno, covar


def pd_write(short_output_pattern, pref_list):
    import pandas as pd
    assert "{0}" in short_output_pattern, "Expect '{0}' in short_output_pattern"

    pref_df = pd.DataFrame(pref_list)
    global output_dir
    output_pattern = output_dir / short_output_pattern
    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        file = Path(str(output_pattern).format(i))
        if not file.exists():
            break
        i += 1
    pref_df.to_csv(file, sep='\t', index=False)
    print(pref_df)


def test_exp_1():
    short_output_pattern ="exp1/exp_1_{0}.tsv"

    # Set these as desired
    os.environ['MKL_THREAD_COUNT'] = '1' #Set this before numpy is imported
    seed = 1
    iid_count = 1*1000 # number of individuals
    sid_count = 5*1000 # number of SNPs
    K0_goal = 200
    leave_out_one_chrom = False

    # Tune these
    proc_count = 1
    GB_goal = 2

    test_snps, pheno, covar = snpsA(seed,iid_count,sid_count)


    pref_list = []
    for use_gpu in [False, True]:
        pref_list.append(test_one_exp(test_snps,K0_goal,seed,pheno,covar,
                                 leave_out_one_chrom=leave_out_one_chrom,use_gpu=use_gpu,proc_count=proc_count,GB_goal=GB_goal))
    pd_write(short_output_pattern,pref_list)

def test_exp_2():
    short_output_pattern ="exp2/exp_2_{0}.tsv"

    # Set these as desired
    os.environ['MKL_THREAD_COUNT'] = '20' #Set this before numpy is imported
    seed = 1 
    iid_count = 10*1000 # number of individuals
    sid_count = 50*1000 # number of SNPs
    K0_goal = 500
    leave_out_one_chrom = False

    # Tune these
    GB_goal = 2

    test_snps, pheno, covar = snpsA(seed,iid_count,sid_count)


    pref_list = []
    for proc_count,use_gpu in [(5,False),(10,False),(1,True),(2,True)]:
        logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
        pref_list.append(test_one_exp(test_snps,K0_goal,seed,pheno,covar,
                                    leave_out_one_chrom=leave_out_one_chrom,use_gpu=use_gpu,proc_count=proc_count,GB_goal=GB_goal))
        pd_write(short_output_pattern+".temp",pref_list)
    pd_write(short_output_pattern,pref_list)

def test_exp_3(K0_goal = 500,GB_goal = 2):
    short_output_pattern = f"exp3/exp_3_{K0_goal}_{'{0}'}.tsv"

    # Set these as desired
    os.environ['MKL_THREAD_COUNT'] = '20' #Set this before numpy is imported
    seed = 1 
    iid_count = 10*1000 # number of individuals
    sid_count = 50*1000 # number of SNPs
    leave_out_one_chrom = True

    # Tune these   

    test_snps, pheno, covar = snpsA(seed,iid_count,sid_count)


    pref_list = []
    for proc_count,use_gpu in [(1,True),(10,False)]:
        logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
        pref_list.append(test_one_exp(test_snps,K0_goal,seed,pheno,covar,
                                    leave_out_one_chrom=leave_out_one_chrom,use_gpu=use_gpu,proc_count=proc_count,GB_goal=GB_goal))
        pd_write(short_output_pattern+".temp",pref_list)
    pd_write(short_output_pattern,pref_list)

def test_exp_4(K0_goal = 500):
    short_output_pattern =f"exp4/exp_4_{K0_goal}_{0}.tsv"

    # Set these as desired
    os.environ['MKL_THREAD_COUNT'] = '1' #Set this before numpy is imported
    seed = 1 
    iid_count = 10*1000 # number of individuals
    sid_count = 50*1000 # number of SNPs
    
    leave_out_one_chrom = True

    # Tune these
    GB_goal = 2

    test_snps, pheno, covar = snpsA(seed,iid_count,sid_count)


    pref_list = []
    for proc_count,use_gpu in [(1,True),(1,False)]:
        logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
        pref_list.append(test_one_exp(test_snps,K0_goal,seed,pheno,covar,
                                    leave_out_one_chrom=leave_out_one_chrom,use_gpu=use_gpu,proc_count=proc_count,GB_goal=GB_goal))
        pd_write(short_output_pattern+".temp",pref_list)
    pd_write(short_output_pattern,pref_list)

def test_exp_3delme(K0_goal = 500, leave_out_one_chrom = True):
    short_output_pattern = f"exp3delme/exp_3delme_{K0_goal}_{'{0}'}.tsv"

    # Set these as desired
    os.environ['MKL_THREAD_COUNT'] = '20' #Set this before numpy is imported
    seed = 1 
    iid_count = 10*1000 # number of individuals
    sid_count = 50*1000 # number of SNPs

    # Tune these

    test_snps, pheno, covar = snpsA(seed,iid_count,sid_count)


    pref_list = []
    GB_goal = 2
    for repeat_i in [1]:
        for proc_count,use_gpu in [(10,False),(1,True)]:
            logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
            pref_list.append(test_one_exp(test_snps,K0_goal,seed,pheno,covar,
                                        leave_out_one_chrom=leave_out_one_chrom,use_gpu=use_gpu,proc_count=proc_count,GB_goal=GB_goal))
            pd_write(short_output_pattern+".temp",pref_list)
    pd_write(short_output_pattern,pref_list)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    test_exp_3(K0_goal=500)