#!!!cmk keep this file in project?

# Don't import numpy until after threads are set
import os

print(os.environ["PYTHONPATH"])
thread_count = 12
os.environ["MKL_NUM_THREADS"] = str(thread_count)  # Set this before numpy is imported
os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
os.environ["OMP_NUM_THREADS"] = str(thread_count)
os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)
from pathlib import Path
import time
import logging
import pysnptools.util as pstutil
import platform

if False:
    cache_top = Path(r"c:\deldir")
    output_dir = Path(r"c:\users\carlk\OneDrive\Projects\Science\gpu_pref")
else:
    if platform.system() == "Windows":
        cache_top = Path(r"m:\deldir")
        output_dir = Path(r"d:\OneDrive\Projects\Science\gpu_pref")
    else:
        cache_top = Path(r"/home/azureuser/deldir")
        output_dir = Path(r"/home/azureuser/gpu_pref_linux")


def one_experiment(
    test_snps,
    K0_goal,
    seed,
    pheno,
    covar,
    leave_out_one_chrom,
    use_gpu,
    proc_count,
    GB_goal,
    just_one_process=False,
    gpu_weight=1,
    gpu_count=1,
):
    import numpy as np
    from pysnptools.util.mapreduce1.runner import LocalMultiProc
    from unittest.mock import patch
    from fastlmm.association import single_snp

    if K0_goal is not None:
        K0 = test_snps[:, :: test_snps.sid_count // K0_goal]
    else:
        K0 = None

    if proc_count == 1:
        runner = None
        xp = "cupy" if use_gpu > 0 else "numpy"
    else:
        if use_gpu == 0:
            runner = LocalMultiProc(proc_count, just_one_process=just_one_process)
            xp = "numpy"
        else:
            assert gpu_count <= proc_count
            weights = [gpu_weight] * gpu_count + [1] * (proc_count - gpu_count)

            def taskindex_to_environ(taskindex):
                if taskindex < gpu_count:
                    return {"ARRAY_MODULE": "cupy", "GPU_INDEX": str(taskindex)}
                else:
                    return {"ARRAY_MODULE": "numpy"}

            xp = "cupy"
            runner = LocalMultiProc(
                proc_count,
                weights=weights,
                taskindex_to_environ=taskindex_to_environ,
                just_one_process=just_one_process,
            )

    #!!!cmk0 change this to using xp=...
    #!!!cmk0 allow strings in util
    #!!!cmk0 test and fix up test_single_snp so works with array_module enviorn set to cupy
    start_time = time.time()

    results_dataframe = single_snp(
        K0=K0,
        test_snps=test_snps,
        pheno=pheno,
        covar=covar,
        leave_out_one_chrom=leave_out_one_chrom,
        count_A1=False,
        GB_goal=GB_goal,
        runner=runner,
        xp=xp,
    )
    delta_time = time.time() - start_time

    K0_count = test_snps.iid_count if K0 is None else K0.sid_count

    perf_result = {
        "test_snps": str(test_snps),
        "iid_count": test_snps.iid_count,
        "test_sid_count": test_snps.sid_count,
        "K0_count": K0_count,
        "low_rank": 0 if test_snps.iid_count == K0_count else 1,
        "seed": seed,
        "chrom_count": len(np.unique(test_snps.pos[:, 0])),
        "covar_count": covar.col_count,
        "leave_out_one_chrom": 1 if leave_out_one_chrom else 0,
        "num_threads": os.environ["MKL_NUM_THREADS"],
        "use_gpu": use_gpu,
        "gpu_weight": gpu_weight,
        "proc_count": proc_count,
        "just_one_process": 1 if just_one_process else 0,
        "GB_goal": GB_goal,
        "time (s)": delta_time,
    }
    return perf_result


def snpsA(seed, iid_count, sid_count):
    import numpy as np
    from pysnptools.snpreader import Bed
    from pysnptools.snpreader import SnpGen

    chrom_count = 10
    global top_cache
    test_snp_path = (
        cache_top / f"snpsA_{seed}_{chrom_count}_{iid_count}_{sid_count}.bed"
    )
    count_A1 = False
    if not test_snp_path.exists():
        snpgen = SnpGen(
            seed=seed,
            iid_count=iid_count,
            sid_count=sid_count,
            chrom_count=chrom_count,
            block_size=1000,
        )
        test_snps = Bed.write(
            str(test_snp_path), snpgen.read(dtype="float32"), count_A1=count_A1
        )
    else:
        test_snps = Bed(str(test_snp_path), count_A1=count_A1)
    from pysnptools.snpreader import SnpData

    np.random.seed(seed)
    pheno = SnpData(
        iid=test_snps.iid,
        sid=["pheno"],
        val=np.random.randn(test_snps.iid_count, 1) * 3 + 2,
    )
    covar = SnpData(
        iid=test_snps.iid,
        sid=["covar1", "covar2"],
        val=np.random.randn(test_snps.iid_count, 2) * 2 - 3,
    )
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
    pref_df.to_csv(file, sep="\t", index=False)
    print(pref_df)
    print(file)


def test_exp_1():
    short_output_pattern = "exp1/exp_1_{0}.tsv"

    # Set these as desired
    os.environ["MKL_NUM_THREADS"] = "1"  # Set this before numpy is imported
    seed = 1
    iid_count = 1 * 1000  # number of individuals
    sid_count = 5 * 1000  # number of SNPs
    K0_goal = 200
    leave_out_one_chrom = False

    # Tune these
    proc_count = 1
    GB_goal = 2

    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count)

    pref_list = []
    for use_gpu in [False, True]:
        pref_list.append(
            one_experiment(
                test_snps,
                K0_goal,
                seed,
                pheno,
                covar,
                leave_out_one_chrom=leave_out_one_chrom,
                use_gpu=use_gpu,
                proc_count=proc_count,
                GB_goal=GB_goal,
            )
        )
    pd_write(short_output_pattern, pref_list)


def test_exp_2():
    short_output_pattern = "exp2/exp_2_{0}.tsv"

    # Set these as desired
    os.environ["MKL_NUM_THREADS"] = "20"  # Set this before numpy is imported
    seed = 1
    iid_count = 10 * 1000  # number of individuals
    sid_count = 50 * 1000  # number of SNPs
    K0_goal = 500
    leave_out_one_chrom = False

    # Tune these
    GB_goal = 2

    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count)

    pref_list = []
    for proc_count, use_gpu in [(5, False), (10, False), (1, True), (2, True)]:
        logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
        pref_list.append(
            one_experiment(
                test_snps,
                K0_goal,
                seed,
                pheno,
                covar,
                leave_out_one_chrom=leave_out_one_chrom,
                use_gpu=use_gpu,
                proc_count=proc_count,
                GB_goal=GB_goal,
            )
        )
        pd_write(short_output_pattern + ".temp", pref_list)
    pd_write(short_output_pattern, pref_list)


def test_exp_3(K0_goal=500, GB_goal=2):
    short_output_pattern = f"exp3/exp_3_{K0_goal}_{'{0}'}.tsv"

    # Set these as desired
    os.environ["MKL_NUM_THREADS"] = "20"  # Set this before numpy is imported
    seed = 1
    iid_count = 10 * 1000  # number of individuals
    sid_count = 50 * 1000  # number of SNPs
    leave_out_one_chrom = True

    # Tune these

    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count)

    pref_list = []
    for proc_count, use_gpu in [(1, True), (10, False)]:
        logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
        pref_list.append(
            one_experiment(
                test_snps,
                K0_goal,
                seed,
                pheno,
                covar,
                leave_out_one_chrom=leave_out_one_chrom,
                use_gpu=use_gpu,
                proc_count=proc_count,
                GB_goal=GB_goal,
            )
        )
        pd_write(short_output_pattern + ".temp", pref_list)
    pd_write(short_output_pattern, pref_list)


def test_exp_4x(K0_goal=500):
    short_output_pattern = f"exp4x/exp_4x_{K0_goal}_{0}.tsv"

    # Set these as desired
    os.environ["MKL_NUM_THREADS"] = "1"  # Set this before numpy is imported
    seed = 1
    iid_count = 10 * 1000  # number of individuals
    sid_count = 50 * 1000  # number of SNPs

    leave_out_one_chrom = True

    # Tune these
    GB_goal = 2

    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count)

    pref_list = []
    for proc_count, use_gpu in [(1, True), (1, False)]:
        logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
        pref_list.append(
            one_experiment(
                test_snps,
                K0_goal,
                seed,
                pheno,
                covar,
                leave_out_one_chrom=leave_out_one_chrom,
                use_gpu=use_gpu,
                proc_count=proc_count,
                GB_goal=GB_goal,
            )
        )
        pd_write(short_output_pattern + ".temp", pref_list)
    pd_write(short_output_pattern, pref_list)


def test_exp_3delme(K0_goal=500, leave_out_one_chrom=True):
    short_output_pattern = f"exp3delme/exp_3delme_{K0_goal}_{'{0}'}.tsv"

    # Set these as desired
    os.environ["MKL_NUM_THREADS"] = "20"  # Set this before numpy is imported
    seed = 1
    iid_count = 10 * 1000  # number of individuals
    sid_count = 50 * 1000  # number of SNPs

    # Tune these

    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count)

    pref_list = []
    GB_goal = 2
    for repeat_i in [1]:
        for proc_count, use_gpu in [(10, False), (1, True)]:
            logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
            pref_list.append(
                one_experiment(
                    test_snps,
                    K0_goal=K0_goal,
                    seed=seed,
                    phen=pheno,
                    covar=covar,
                    leave_out_one_chrom=leave_out_one_chrom,
                    use_gpu=use_gpu,
                    proc_count=proc_count,
                    GB_goal=GB_goal,
                )
            )
            pd_write(short_output_pattern + ".temp", pref_list)
    pd_write(short_output_pattern, pref_list)


def test_std():
    from pysnptools.standardizer.standardizer import Standardizer

    seed = 1
    iid_count = 2 * 1000
    sid_count = 50 * 1000  # number of SNPs
    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count)
    xp = pstutil.array_module_from_env("cupy")
    print(xp.__name__)
    snps = xp.asarray(test_snps.read().val)
    stats = xp.empty(
        [snps.shape[1], 2],
        dtype=snps.dtype,
        order="F" if snps.flags["F_CONTIGUOUS"] else "C",
    )

    for i in range(15):
        Standardizer._standardize_unit_python(
            snps, apply_in_place=True, use_stats=False, stats=stats
        )
    print("done!")


def test_exp_4(
    GB_goal=2,
    iid_count=2 * 1000,
    K0_goal=None,
    proc_count_only_cpu=10,
    proc_count_with_gpu=5,
    gpu_weight=3,
    gpu_count=1,
    num_threads=20,
    leave_out_one_chrom=True,
    just_one_process=False,
):
    short_output_pattern = f"exp4/exp_4_{'{0}'}.tsv"

    # Set these as desired
    os.environ["MKL_NUM_THREADS"] = str(
        num_threads
    )  # Set this before numpy is imported
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    seed = 1
    sid_count = 50 * 1000  # number of SNPs

    # Tune these

    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count)

    proc_gpu_list = []
    if proc_count_only_cpu > 0:
        proc_gpu_list += [(proc_count_only_cpu, 0)]
    if proc_count_with_gpu > 0:
        proc_gpu_list += [(proc_count_with_gpu, 0.5)]
    proc_gpu_list += [(1, 1)]
    pref_list = []
    for proc_count, use_gpu in proc_gpu_list:
        logging.info(f"proc_count={proc_count},use_gpu={use_gpu}")
        pref_list.append(
            one_experiment(
                test_snps,
                K0_goal=K0_goal,
                seed=seed,
                pheno=pheno,
                covar=covar,
                leave_out_one_chrom=leave_out_one_chrom,
                use_gpu=use_gpu,
                proc_count=proc_count,
                GB_goal=GB_goal,
                just_one_process=just_one_process,
                gpu_weight=gpu_weight,
                gpu_count=gpu_count,
            )
        )
        pd_write(short_output_pattern + ".temp", pref_list)
    pd_write(short_output_pattern, pref_list)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARN)
    test_exp_4(
        GB_goal=4,
        iid_count=5 * 1000,
        K0_goal=2000,
        proc_count_only_cpu=12,
        proc_count_with_gpu=2,
        gpu_weight=9,
        gpu_count=2,
        num_threads=12,
        leave_out_one_chrom=True,
        just_one_process=False,
    )
    # test_std()

