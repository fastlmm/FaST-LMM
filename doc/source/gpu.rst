GPU association and floating-point precision
============================================

These features are under development on the GPU branches; installing the latest
published package does not necessarily provide them. The examples below target
this branch's API.

Run a small association study
-----------------------------

``single_snp`` accepts ``xp="numpy"`` or ``xp="cupy"`` and
``dtype="float64"`` (the default) or ``dtype="float32"``. NumPy runs on the CPU;
CuPy requires a working GPU installation. A missing CuPy installation raises
``ImportError``; requesting it does not silently select NumPy. Import CuPy
explicitly and check the device before requesting it. A successful import alone
does not verify that the driver can execute GPU operations.

This complete example creates deterministic data in memory, runs the default
leave-one-chromosome-out (LOCO) analysis on both devices, and aligns results by
SNP before comparing them::

    import cupy as cp
    import numpy as np
    from fastlmm.association import single_snp
    from pysnptools.snpreader import SnpData

    assert cp.cuda.runtime.getDeviceCount() > 0
    cp.zeros(1).sum().item()  # also exercise a GPU operation
    rng = np.random.default_rng(7)
    iid = [["family", str(sample)] for sample in range(40)]
    sid = [f"snp{marker}" for marker in range(60)]
    genotypes = rng.integers(0, 3, size=(40, 60)).astype(np.float64)
    positions = [[1 + marker // 30, 0, marker + 1] for marker in range(60)]
    snps = SnpData(iid=iid, sid=sid, val=genotypes, pos=positions, xp="numpy")
    phenotype = SnpData(
        iid=iid, sid=["trait"],
        val=(0.3 * genotypes[:, 0] + rng.normal(size=40))[:, None], xp="numpy"
    )
    cpu = single_snp(snps, phenotype, xp="numpy", dtype="float64")
    gpu = single_snp(snps, phenotype, xp=cp, dtype="float32")
    comparison = cpu.set_index("SNP")[["PValue"]].join(
        gpu.set_index("SNP")[["PValue"]], lsuffix="_cpu64", rsuffix="_gpu32"
    )
    assert len(comparison) == len(sid) and comparison.notna().all().all()
    print(comparison.head())

Both calls return a pandas DataFrame on the CPU, sorted by p-value. Float32 and
GPU results need not be identical to CPU float64 results. Check differences in
p-values and fitted parameters on representative data before choosing precision;
there is no universal tolerance for every study. To isolate the effect of
precision, also run the CPU with ``dtype="float32"``.

Choose precision and memory use
-------------------------------

Float32 stores the main LMM covariates, phenotypes, kernel/factor, decomposition,
and test SNP arrays in single precision. On the CPU, training kernel/factor
formation still uses float64 before conversion to the requested LMM dtype; GPU
formation uses the requested dtype. Per-SNP statistics use float64 arithmetic, but
this cannot recover precision already lost in the decomposition or rotations.
Float32 uses less array memory and can accelerate consumer GPUs with limited
float64 throughput. GPUs with strong float64 support may perform well with the
default precision. Timings depend on hardware, problem size, and transfers.

Covariate or phenotype values outside the safe float32 range raise
``ValueError``; rescale them or use float64. Only float32 and float64 are
accepted. ``GB_goal`` controls blocking decisions and is a memory target, not a
hard limit on GPU allocations: kernels, solver workspaces, and allocator pools
also consume memory. Low-rank analysis still needs the full genotype factor;
blocking does not make every step fit an arbitrarily small device.

Read BED data on the GPU
------------------------

With a GPU-enabled bed-reader build (the ``cubecl-cuda`` Cargo feature) and CuPy,
this branch can read and Unit-standardize BED blocks on the GPU and form the
kernel or low-rank factor there. The test-SNP path derives the variance needed for ``EffectSize`` from Unit
training statistics, avoiding a second genotype read. It squares the trained
standard deviation in float64 and maps the constant-SNP infinity sentinel to
zero; float32 training moments still carry float32 rounding.
Selecting CuPy alone does not enable native GPU BED decoding. A BED GPU request
requires a CUDA-enabled bed-reader build and raises if native GPU support is
missing. Other input formats may read on the CPU and transfer values to CuPy;
custom standardizers must support the selected array device.

This independent example writes its own BED/BIM/FAM files and performs LOCO.
It observes the actual CuPy array passed to the GPU eigensolver, rather than
inferring GPU execution from the pandas result::

    import cupy as cp
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from bed_reader import has_gpu_support, to_bed
    from fastlmm.association import single_snp
    from pysnptools.snpreader import Bed, SnpData

    if not has_gpu_support():
        raise RuntimeError("this example requires a bed-reader build with the cubecl-cuda Cargo feature")
    rng = np.random.default_rng(23)
    values = rng.integers(0, 3, size=(40, 60)).astype(np.float64)
    observed = []
    original_eigh = cp.linalg.eigh

    def observe_eigh(matrix, *args, **kwargs):
        assert isinstance(matrix, cp.ndarray) and matrix.dtype == np.float32
        observed.append(matrix.shape)
        return original_eigh(matrix, *args, **kwargs)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "cohort.bed"
        to_bed(path, values, properties={"chromosome": [1] * 30 + [2] * 30},
               count_A1=True)
        bed = Bed(path, count_A1=True)
        phenotype = SnpData(iid=bed.iid, sid=["trait"],
                            val=rng.normal(size=(40, 1)), xp=np)
        with patch.object(cp.linalg, "eigh", observe_eigh):
            results = single_snp(bed, phenotype, xp=cp, dtype="float32",
                                 force_full_rank=True)
        assert isinstance(results, pd.DataFrame) and len(results) == bed.sid_count
        assert observed and results.PValue.between(0, 1).all()

Keep ``count_A1`` explicit so allele coding is clear. LOCO is enabled by default;
using ``leave_out_one_chrom=False`` can cause proximal contamination.

Backend scope and limitations
-----------------------------

An explicit ``xp`` overrides ``ARRAY_MODULE``; otherwise the environment selects
the backend, with NumPy as the final default. ``single_snp`` temporarily sets
``ARRAY_MODULE`` for the workflow and restores it afterward. Treat this as a
process-wide setting when arranging concurrent workflows and runners; configure
worker environments consistently. Backend selection, execution, and restoration
are serialized between ``single_snp`` calls within one process. An omitted ``xp``
resolves after preceding calls restore their environment. The lock does not
coordinate unrelated code changing ``os.environ``; explicit ``xp`` keeps the
workflow choice clear. Standalone reader behavior is described in
PySnpTools's GPU guide.

Two-kernel analyses (``K1`` or ``G1``) run on NumPy. A requested GPU backend
produces a warning in this case. This guide covers ``single_snp``; it does not
promise GPU or float32 support for every FaST-LMM entry point.

Reuse precomputation safely
---------------------------

A cache requires the same phenotype, covariates, and training inputs; it is not
a general cache keyed by input contents. Use distinct paths for distinct
analyses and precisions. A cache with another dtype raises ``ValueError``.
This independent CPU example changes a supplied heritability while reusing the
same decomposition and verifies it against an uncached calculation::

    import numpy as np
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from fastlmm.association import single_snp
    from pysnptools.snpreader import SnpData

    rng = np.random.default_rng(29)
    iid = [["family", str(sample)] for sample in range(30)]
    sid = [f"snp{marker}" for marker in range(20)]
    snps = SnpData(iid=iid, sid=sid,
                   val=rng.integers(0, 3, size=(30, 20)).astype(np.float64), xp=np)
    phenotype = SnpData(iid=iid, sid=["trait"], val=rng.normal(size=(30, 1)), xp=np)

    with TemporaryDirectory() as directory:
        cache = str(Path(directory) / "cpu64.npz")
        first = single_snp(snps, phenotype, xp="numpy", dtype="float64",
                           h2=0.5, cache_file=cache, leave_out_one_chrom=False)
        second = single_snp(snps, phenotype, xp="numpy", dtype="float64",
                            h2=0.6, cache_file=cache, leave_out_one_chrom=False)
        fresh = single_snp(snps, phenotype, xp="numpy", dtype="float64", h2=0.6,
                           leave_out_one_chrom=False)
        assert len(first) == len(second) == len(sid)
        np.testing.assert_allclose(second.sort_values("SNP").PValue,
                                   fresh.sort_values("SNP").PValue, rtol=1e-12, atol=0)

An explicitly supplied ``h2`` takes precedence when reusing the decomposition.
If a cache was created with a supplied ``h2`` and a later call omits it, the later
call fits ``h2``. Multiple phenotypes can use a supplied ``h2`` too. Changing an
explicit ``mixing`` to disagree with the cached mixing raises ``ValueError``;
create a separate cache for that model. LOCO creates a separate cache per
chromosome, and multiple phenotypes can create an additional ``.extra.npz`` file.

Reader API and worker execution
-------------------------------

BED GPU reads and standardization are implemented by PySnpTools's reader API.
The ``single_snp(..., xp=..., dtype=...)`` call remains the public FaST-LMM
workflow; callers do not need private native-reader helpers. Standalone
PySnpTools readers accept explicit ``xp`` arguments, and standardization follows
the existing array's device.

``single_snp`` establishes the workflow's ``ARRAY_MODULE`` and restores it on
success or failure. Overlapping calls within one process are serialized to
protect that process-wide context. Runner workers resolve their backend from
their own environment, including explicit runner overrides; model arrays move to
the worker's backend. A parent CuPy call alone therefore does not prove every
worker ran on the GPU. Two-kernel analyses continue to use NumPy.

These APIs are present in the development checkout. Published dependency bounds
and acceptance against published artifacts still require coordinated release
work.


This complete worker example checks array types inside real subprocesses and
then uses the same runner for association. The example runs in its own temporary
working directory because LocalMultiProc writes ``runs/`` files relative to the
working directory. Keep ``xp`` explicit at the public workflow boundary::

    import os
    import cupy as cp
    import numpy as np
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from bed_reader import has_gpu_support, to_bed
    from fastlmm.association import single_snp
    from pysnptools.snpreader import Bed, SnpData
    from pysnptools.util import get_array_module
    from pysnptools.util.mapreduce1 import map_reduce
    from pysnptools.util.mapreduce1.runner import LocalMultiProc

    if not has_gpu_support():
        raise RuntimeError("this example requires a bed-reader build with the cubecl-cuda Cargo feature")

    def worker_environment(task_index):
        return {"ARRAY_MODULE": "cupy" if task_index % 2 == 0 else "numpy"}

    rng = np.random.default_rng(31)
    with TemporaryDirectory() as directory:
        previous = Path.cwd()
        os.chdir(directory)
        try:
            path = Path(directory) / "cohort.bed"
            values = rng.integers(0, 3, size=(40, 24)).astype(np.float64)
            to_bed(path, values, count_A1=True,
                   properties={"chromosome": [1] * 12 + [2] * 12})
            bed = Bed(path, count_A1=True)
            phenotype = SnpData(iid=bed.iid, sid=["trait"],
                                val=rng.normal(size=(40, 1)), xp=np)
            runner = LocalMultiProc(2, taskindex_to_environ=worker_environment)

            def read_backend(task_index):
                data = bed[:, task_index:task_index + 1].read(dtype="float32")
                actual = get_array_module(data.val).__name__
                assert actual == os.environ["ARRAY_MODULE"]
                assert data.val.dtype == np.float32
                return actual, os.getpid()

            observations = map_reduce(range(2), mapper=read_backend,
                                      reducer=list, runner=runner)
            assert {backend for backend, process in observations} == {"numpy", "cupy"}
            assert all(process != os.getpid() for backend, process in observations)
            with patch.dict("os.environ", {"ARRAY_MODULE": "before-call"}):
                results = single_snp(bed, phenotype, xp=cp, runner=runner,
                                     h2=0.3, dtype="float32", force_full_rank=True)
                assert os.environ["ARRAY_MODULE"] == "before-call"
                try:
                    single_snp(None, None, xp=np)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("missing test SNPs must fail")
                assert os.environ["ARRAY_MODULE"] == "before-call"
            assert len(results) == bed.sid_count
        finally:
            os.chdir(previous)

This verifies the worker read arrays directly. The association call returns
CPU pandas results even when some workers use CuPy. Runner environment callbacks
can also be applied to a reducer task, so return a valid mapping for every task
index. Both inner SNP scheduling and outer chromosome scheduling use worker
backend resolution; supplying a parent CuPy array does not fix worker placement.

Options and results
-------------------

.. list-table:: Important ``single_snp`` options
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Behavior
   * - ``xp``
     - ``None``
     - Explicit NumPy/CuPy module or name overrides ``ARRAY_MODULE``; otherwise
       the environment selects, with NumPy as the final default. Missing CuPy
       raises ``ImportError``; an unknown name raises ``ValueError``.
   * - ``dtype``
     - ``"float64"``
     - Float32/float64 only. An unsupported floating dtype or unsafe finite
       float32 covariate/phenotype range raises ``ValueError``. Float32 range
       checks do not establish numerical accuracy for a study.
   * - ``leave_out_one_chrom``
     - ``True``
     - LOCO. ``covar_by_chrom`` requires this mode. ``K0`` may map chromosomes
       to precomputed kernels; each must exclude that chromosome appropriately.
   * - ``K0``, ``K1`` / ``G0``, ``G1``
     - ``None``
     - ``K0`` defaults to test SNPs. ``G0``/``G1`` are compatibility aliases.
       With a second kernel, execution uses NumPy; a requested GPU warns.
       Multiple phenotypes are not supported with a second kernel.
   * - ``h2``, ``mixing``
     - ``None``
     - Fit when unspecified. A supplied scalar ``h2`` is repeated for all
       phenotypes; an array supplies one per phenotype. When fitting a
       two-kernel mixing weight, ``h2`` must also be unspecified.
   * - ``cache_file``
     - ``None``
     - Optional decomposition/moment cache. Check inputs yourself; there is
       no content fingerprint. Dtype or explicit mixing mismatch raises.
   * - ``GB_goal``
     - ``None``
     - A blocking target, not a hard device memory cap. Blocking estimates
       still assume eight bytes per value, including float32 runs.
   * - ``force_full_rank``, ``force_low_rank``
     - Both ``False``
     - Optional rank strategy. Setting both raises ``Exception``. A low-rank
       factor must still fit in memory; a full-rank kernel is samples squared.
   * - ``runner``, ``map_reduce_outer``
     - ``None``, ``True``
     - Local execution by default. Outer work is by chromosome when LOCO is
       enabled; ``False`` schedules test-SNP chunks. Workers use their own
       ``ARRAY_MODULE``, including runner overrides.
   * - ``count_A1``
     - ``None``
     - Used when creating BED readers from paths. An already-constructed
       ``Bed`` controls its own allele counting. Omission currently means
       A2 counting with a warning when a BED reader is constructed.
   * - ``show_snp_fract_var_exp``
     - ``False``
     - Adds ``SnpFractVarExpl``. ``EffectSize`` is always returned.
   * - ``pvalue_threshold``, ``random_threshold``
     - ``None``
     - Optional row filters; ``random_seed=0`` controls random row sampling.
       Results need not contain every input SNP when filters are enabled.

The return is a CPU pandas DataFrame sorted by ``PValue``. With multiple
phenotypes it has a row per retained SNP/phenotype pair and a ``Pheno`` column.
Other columns include ``SNP``, ``sid_index``, ``Chr``, ``GenDist``, ``ChrPos``,
``PValue``, ``SnpWeight``, ``SnpWeightSE``, ``EffectSize``, ``Mixing``, and
``Nullh2``. An ``output_file_name`` also writes tab-delimited results.
Array memory order is an internal choice here; ``single_snp`` has no ``order``
argument and does not return a device array.

Native BED GPU buffers have a ``2**32 - 1`` element indexing limit plus further
standardization limits. Device memory is usually more restrictive. Both native
and CuPy pools, temporary products, the factor/kernel, and solver workspace
contribute; CUDA allocation failures propagate. Python HIP is not implemented.
CPU kernel formation and its intermediate memory can remain float64 even for a
float32 LMM. No timing or memory guarantee follows from ``dtype`` alone.
