#!!! keep this file in project?

# Don't import numpy until after threads are set
import os

if False:
    thread_count = 12
    os.environ["MKL_NUM_THREADS"] = str(
        thread_count
    )  # Set this before numpy is imported
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)
from pathlib import Path
import time
import logging
import pysnptools.util as pstutil
import platform
import multiprocessing

if False:
    cache_top = Path(r"c:\deldir")
    output_dir = Path(r"c:\users\carlk\OneDrive\Projects\Science\gpu_pref")
else:
    if platform.system() == "Windows":
        cache_top = Path(r"m:\deldir")
        output_dir = Path(r"d:\OneDrive\Projects\Science\gpu_pref")
    else:
        if "WSL_DISTRO_NAME" in os.environ:
            cache_top = Path(r"/mnt/m/deldir")
            output_dir = Path(r"/mnt/d/OneDrive\Projects\Science\gpu_pref_linux")
        else:
            cache_top = Path("/home/azureuser/deldir")
            output_dir = Path("/home/azureuser/gpu_pref_linux")


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
    cpu_weight=1,
    gpu_weight=1,
    gpu_count=1,
    test_case="?",
    force = None,
    two_ks = False,
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
            weights = [gpu_weight] * gpu_count + [cpu_weight] * (proc_count - gpu_count)

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

    force_full_rank = False
    force_low_rank = False
    if force == "full_rank":
        force_full_rank = True
    elif force == "low_rank":
        force_low_rank = True
    elif force is None:
        pass
    else:
        assert False

    start_time = time.time()

    results_dataframe = single_snp(
        K0=K0,
        K1 = test_snps if two_ks else None,
        test_snps=test_snps,
        pheno=pheno,
        covar=covar,
        leave_out_one_chrom=leave_out_one_chrom,
        count_A1=False,
        GB_goal=GB_goal,
        runner=runner,
        xp=xp,
        force_full_rank=force_full_rank,
        force_low_rank=force_low_rank,
    )
    delta_time = time.time() - start_time

    K0_count = test_snps.iid_count if K0 is None else K0.sid_count

    perf_result = {
        "computer_name": os.environ.get("COMPUTERNAME", "<unknown>"),
        "cpu_count": multiprocessing.cpu_count(),
        "gpu_count": "1?",
        "test_case": test_case,
        "linked": "MKL?/OpenBLAS?",
        "test_snps": str(test_snps),
        "iid_count": test_snps.iid_count,
        "test_sid_count": test_snps.sid_count,
        "low_rank": 0 if test_snps.iid_count == K0_count else 1,
        "K0_count": K0_count,
        "seed": seed,
        "chrom_count": len(np.unique(test_snps.pos[:, 0])),
        "covar_count": covar.col_count,
        "leave_out_one_chrom": 1 if leave_out_one_chrom else 0,
        "num_threads": os.environ.get("MKL_NUM_THREADS", "<none>"),
        "use_gpu": use_gpu,
        "cpu_weight": cpu_weight,
        "gpu_weight": gpu_weight,
        "proc_count": proc_count,
        "just_one_process": 1 if just_one_process else 0,
        "GB_goal": GB_goal,
        "time (s)": delta_time,
        "force": force if force is not None else "<none>",
        "two_ks": 1 if two_ks else 0,
    }
    return perf_result


def snpsA(seed, iid_count, sid_count, use_distributed):
    import numpy as np
    from pysnptools.snpreader import Bed
    from pysnptools.snpreader import DistributedBed
    from pysnptools.snpreader import SnpGen

    chrom_count = 10
    global top_cache
    if use_distributed:
        test_snp_path = (
            cache_top / f"snpsA_{seed}_{chrom_count}_{iid_count}_{sid_count}_db"
        )
    else:
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
        if use_distributed:
            test_snps = DistributedBed.write(str(test_snp_path), snpgen)
        else:
            test_snps = Bed.write(
                str(test_snp_path), snpgen.read(dtype="float32"), count_A1=count_A1
            )
    else:
        if use_distributed:
            test_snps = DistributedBed(str(test_snp_path))
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


def testX_exp_4x(K0_goal=500):
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





def test_std():
    from pysnptools.standardizer.standardizer import Standardizer

    seed = 1
    iid_count = 2 * 1000
    sid_count = 50 * 1000  # number of SNPs
    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count)
    xp = pstutil.array_module("cupy")
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
    sid_count=50 * 1000,
    K0_goal=None,
    proc_count_only_cpu=10,
    proc_count_with_gpu=5,
    cpu_weight=1,
    gpu_weight=1,
    gpu_count=1,
    num_threads=None,
    leave_out_one_chrom=True,
    just_one_process=False,
    force = None,
    two_ks = False,
    use_distributed = False,
):
    short_output_pattern = f"exp4/exp_4_{'{0}'}.tsv"

    # Set these as desired
    if num_threads is not None:
        os.environ["MKL_NUM_THREADS"] = str(
            num_threads
        )  # Set this before numpy is imported
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    seed = 1

    # Tune these

    test_snps, pheno, covar = snpsA(seed, iid_count, sid_count,use_distributed=use_distributed)

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
                cpu_weight=cpu_weight,
                gpu_weight=gpu_weight,
                gpu_count=gpu_count,
                force = force,
                two_ks = two_ks,
            )
        )
        pd_write(short_output_pattern + ".temp", pref_list)
    pd_write(short_output_pattern, pref_list)


def test_case_def(test_case):
    if test_case == "a":
        iid_count = 2000
        sid_count = 50 * 1000
        K0_goal = None
    elif test_case == "b":
        iid_count = 4000
        sid_count = 50 * 1000
        K0_goal = None
    elif test_case == "c":
        iid_count = 5000
        sid_count = 50 * 1000
        K0_goal = 2000
    elif test_case == "d":
        iid_count = 1000
        sid_count = 100 * 1000
        K0_goal = 1000
    elif test_case == "e":
        iid_count = 2000
        sid_count = 1000 * 1000
        K0_goal = None
    elif test_case == "f":
        iid_count = 200
        sid_count = 200
        K0_goal = None
    else:
        assert False, test_case

    return test_case, iid_count, sid_count, K0_goal


def test_svd(size, which_list, threads=None):
    # may want to play with "F" vs "C" order
    if threads is not None:
        os.environ["MKL_NUM_THREADS"] = str(threads)
    import numpy as np
    from fastlmm.util.matrix.bigsvd import big_sdd, lapack_svd

    short_output_pattern = f"svd/svd_{'{0}'}.tsv"
    which_list = which_list or [
        "big_sdd",
        "lapack_svd",
        "np.linalg.svd",
        "cp.linalg.svd",
    ]
    m_row = size
    n_col = m_row + 2
    min_row_col = size
    perf_list = []
    for which in which_list:
        logging.info("generating {0}x{1}".format(m_row, n_col))
        np.random.seed(0)
        a = np.random.randn(m_row, n_col).astype(
            np.float
        )  # what's the "float about?"
        if which == "big_sdd":
            logging.info("doing large big_sdd")
            start_time = time.time()
            # _ = lapack_svd(np.array([[1],[-2],[3]],order="F"))
            ux, sx, vtx = big_sdd(
                np.array(a, order="F"),
                work_around=True
            ) 
            delta_time = time.time() - start_time
            logging.info(f"done with big_sdd {m_row}x{n_col} in time {delta_time}")
            Sx = np.zeros((m_row, n_col))
            Sx[:min_row_col, :min_row_col] = np.diag(sx)
            assert np.allclose(a, np.dot(ux, np.dot(Sx, vtx)))
            xp = np
        elif which == "lapack_svd":
            logging.info("doing lapack_svd")
            start_time = time.time()
            ux, sx, vtx = lapack_svd(np.array(a, order="F"))
            delta_time = time.time() - start_time
            logging.info(
                f"done with xp.linalg.svd {m_row}x{n_col} in time {delta_time}"
            )
            Sx = np.zeros((m_row, n_col))
            Sx[:min_row_col, :min_row_col] = np.diag(sx)
            assert np.allclose(a, np.dot(ux, np.dot(Sx, vtx)))
            xp = np
        elif which == "np.linalg.svd":
            xp = np
            logging.info(f"Doing large np.linalg.svd ({m_row}x{n_col})")
            start_time = time.time()
            U, s, V = np.linalg.svd(a, full_matrices=False, compute_uv=True)
            delta_time = time.time() - start_time
            logging.info(
                f"done with np.linalg.svd {m_row}x{n_col} in time {delta_time}"
            )
            print(U.shape, s.shape, V.shape, end=" ")
            S = np.zeros((min_row_col, min_row_col))
            S = np.diag(s)
            assert np.allclose(a, np.dot(U, np.dot(S, V)))
        elif which == "cp.linalg.svd":
            import cupy as cp

            xp = cp
            logging.info(f"Doing large cp.linalg.svd ({m_row}x{n_col})")
            a = cp.asarray(a)
            start_time = time.time()
            U, s, V = cp.linalg.svd(a, full_matrices=False, compute_uv=True)
            delta_time = time.time() - start_time
            logging.info(
                f"done with cp.linalg.svd {m_row}x{n_col} in time {delta_time}"
            )
            print(U.shape, s.shape, V.shape, end=" ")
            S = cp.zeros((min_row_col, min_row_col))
            S = cp.diag(s)
            assert cp.allclose(a, cp.dot(U, cp.dot(S, V)))
        else:
            assert False
        perf_result = {
            "computer_name": os.environ.get("COMPUTERNAME", "<unknown>"),
            "cpu_count": multiprocessing.cpu_count(),
            "gpu_count": "1?",
            "test_case": "test_svd",
            "linked": "MKL?/OpenBLAS?",
            "size": size,
            "which": which,
            "num_threads": os.environ.get("MKL_NUM_THREADS", "<none>"),
            "xp": xp.__name__,
            "time (s)": delta_time,
        }
        perf_list.append(perf_result)
        pd_write(short_output_pattern + ".temp", perf_list)
    pd_write(short_output_pattern, perf_list)

def benchmark1():
    #based on https://towardsdatascience.com/heres-how-to-use-cupy-to-make-numpy-700x-faster-4b920dda1f56
    import numpy as np
    import cupy as cp
    import time

    ### Numpy and CPU
    s = time.time()
    x_cpu = np.ones((500,1000,1000))
    e = time.time()
    print(e - s)

    ### CuPy and GPU
    s = time.time()
    x_gpu = cp.ones((500,1000,1000))
    #cp.cuda.Stream.null.synchronize()
    e = time.time()
    print(e - s)

    ### Numpy and CPU
    s = time.time()
    x_cpu *= 5
    e = time.time()
    print(e - s)
    ### CuPy and GPU
    s = time.time()
    x_gpu *= 5
    cp.cuda.Stream.null.synchronize()
    e = time.time()
    print(e - s)

    print("done")

def benchmark2():
    #Based on http://learningsys.org/nips17/assets/papers/paper_16.pdf
    import time, numpy
    #import cupy

    def test1(xp,size,repeat):
        a = xp.arange(size).reshape(1000, -1)
        #cupy.cuda.runtime.deviceSynchronize()
        t1 = time.time()
        for index in range(repeat):
            a = a.T * 2
        #cupy.cuda.runtime.deviceSynchronize()
        t2 = time.time()
        diff = t2-t1
        print(f"{repeat}x{size:,} {xp.__name__} {diff}")
        return max(t2-t1,1e-16)

    def test(xp,size,repeat):
        a = xp.arange(size).reshape(1000, -1)
        b = xp.arange(size).reshape(-1, 1000)
        #cupy.cuda.runtime.deviceSynchronize()
        t1 = time.time()
        for index in range(repeat):
            a = xp.dot(b,a)
        #cupy.cuda.runtime.deviceSynchronize()
        t2 = time.time()
        diff = t2-t1
        print(f"{repeat}x{size:,} {xp.__name__} {diff}")
        return max(t2-t1,1e-16)


    repeat = 50
    for exp in range(6,9):
        size = 10 ** exp
        t_dict = {}
        for xp in [numpy]:
             t_dict[xp.__name__+"1st"] = test(xp,size,1) # Avoid first call overhead
             t_dict[xp.__name__+"2nd"] = test(xp,size,1)
        #print(f"{repeat}x{size:,} 1st:{t_dict['numpy1st']/t_dict['cupy1st']} 2nd:{t_dict['numpy2nd']/t_dict['cupy2nd']}")
    # (’numpy’, 0.5105748176574707)
    # (’cupy’, 0.08418107032775879)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    if True:
        test_case, iid_count, sid_count, K0_goal = test_case_def("d")

        for two_ks in [False]:
            for force in ["full_rank","low_rank"]:
                test_exp_4(
                    GB_goal=4,
                    iid_count=iid_count,
                    K0_goal=K0_goal,
                    proc_count_only_cpu=0,
                    proc_count_with_gpu=6,
                    cpu_weight=4,
                    gpu_weight=3,
                    gpu_count=2,
                    num_threads=None,
                    leave_out_one_chrom=True,
                    just_one_process=False,
                    force = force,
                    two_ks = two_ks,
                    use_distributed=True
            )

    #benchmark2()

    #test_svd(3_000, which_list=None)

