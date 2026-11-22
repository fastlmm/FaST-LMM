GPU association and floating-point precision
============================================

This guide describes the ``1.1.0a1`` alpha. NumPy remains the default backend.
This development GPU backend requires the upcoming PySnpTools release with
CPU-only bed-reader support; published PySnpTools 1.1.0a1 uses the former GPU API.

Installing development GPU support
-----------------------------------

Install the CPU-only bed-reader beta wheel, then the matching development
checkouts into one environment::

    uv pip install "/path/to/PySnpTools[cupy]" "/path/to/FaST-LMM[cupy]"

.. note::

   Published ``1.1.0a1`` packages use bed-reader's former GPU API and do not
   include the numerical fixes and API policy described below. The revised
   notebook also requires PySnpTools' development ``num_threads`` support.
   Coordinated published-artifact qualification remains necessary before release.

The ``cupy`` extra follows the dependency chain through ``pysnptools[cupy]``
and CuPy directly. Follow
`CuPy's installation guide <https://docs.cupy.dev/en/stable/install.html>`_
for CUDA driver/toolkit prerequisites. A CPU bed-reader wheel is sufficient.
GPU support targets NVIDIA Windows/Linux x86-64, including WSL2.
CuPy 14.2 has no Python 3.15 wheel, so Python 3.15 is currently CPU-only. Linux GPU wheels require glibc 2.35 or later. The published alpha
stack passed native Linux GPU integration tests on Ubuntu 24.04 x86-64 with
an RTX 3060, Python 3.11, and CUDA toolkit 13.3, in addition to Windows and
WSL2 testing. Other CPU
platforms can use bed-reader's CPU wheels, subject to FaST-LMM's existing
dependency limits; Windows ARM64 requires Python 3.13 or later. Omit ``[cupy]``
for CPU-only use. Normal unpinned installations still
select stable releases.

Set the defaults once for a workflow::

    import os

    os.environ["ARRAY_MODULE"] = "cupy"
    os.environ["ARRAY_DTYPE"] = "float32"

Explicit ``xp`` and ``dtype`` still take precedence. These defaults are
process-wide. Follow the function-specific limitations below; selecting CuPy
for ``single_snp`` does not extend GPU support to every FaST-LMM function.

Float64 arithmetic can be much slower than float32 on consumer GPUs. Some
data-center GPUs have high float64 throughput, so this is device-dependent.
Float64 arrays also use twice the storage. GPU float64 workflows emit
``pysnptools.util.gpu.GpuFloat64Warning``; this includes cached models and GPU workers.
The warning does not change precision or defaults. Choose
``single_snp(..., xp="cupy", dtype="float32")`` only when its precision is adequate,
and compare p-values and effect sizes against float64 on representative data.
Float32 can change results, especially for poorly conditioned problems.

CPU float64 does not trigger this notice. Float32 specifies the main linear
algebra; internal reductions and likelihood statistics deliberately retain
higher precision and do not trigger user-storage warnings. Raw phenotype and
covariate values are validated at full precision on the CPU before narrowing
and GPU transfer, avoiding unnecessary float64 GPU intermediates. Python's default warning filter
avoids repeated identical notices at one user call site. The notice is
``pysnptools.util.gpu.GpuFloat64Warning``. See the dtype documentation below for numerical
safeguards and cache compatibility.

Workflow-wide precision defaults
--------------------------------

An explicit ``single_snp(dtype=...)`` overrides ``ARRAY_DTYPE``. None or omission
uses that environment variable, then float64. Only float32/float64 are accepted.
FaST-LMM establishes the resolved precision alongside ``ARRAY_MODULE`` under its
workflow lock, so nested reads and workers share the default. Both settings are
restored on success or failure. Worker-specific ARRAY_DTYPE overrides must agree
with the workflow precision; conflicting settings raise ValueError.
Conflicting simultaneous workflows remain
serialized; direct environment changes outside the workflow are process-wide.

A standalone NumPy/CuPy function does not read these settings: numerical
allocations still specify dtype where needed. Existing arrays retain their
dtype; standardizers follow those arrays. Kernel and factor formation follow
the workflow dtype on both backends; explicit higher-precision statistics and
the float64 kernel-diagonal sum remain unchanged.

For example, using the existing sample files::

    from unittest.mock import patch
    from fastlmm.association import single_snp
    from fastlmm.util import example_file
    from pysnptools.snpreader import Bed

    snps = Bed(example_file("fastlmm/feature_selection/examples/toydata.5chrom.*", "*.bed"),
               count_A1=False)[:40, :60]
    pheno = example_file("fastlmm/feature_selection/examples/toydata.phe")
    with patch.dict("os.environ", {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}):
        results = single_snp(snps, pheno, leave_out_one_chrom=False)
    # Omitted xp/dtype use the shared defaults. Explicit arguments override them.

This policy is implemented by the PySnpTools and bed-reader versions that
FaST-LMM requires.

Run a small association study
-----------------------------

``single_snp`` accepts ``xp="numpy"`` or ``xp="cupy"`` and
``dtype="float64"`` or ``dtype="float32"``. None or omission uses
``ARRAY_DTYPE``, then float64. NumPy runs on the CPU;
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

Numerical qualification limits
------------------------------

Independent variance-component fits can differ even in float64. Tiny backend
roundoff changes which h2 is selected when likelihood values near the optimum
are indistinguishable at floating-point precision. A strict comparison of
independently fitted p-values therefore also measures optimizer sensitivity.
For a backend arithmetic diagnostic, fit h2 on the CPU and pass that same h2 to
both calls, matching the background kernel and chromosome exclusions. Retain
both the independent-fit results and this diagnostic; passing the latter does
not make the former pass.
An independent long-double likelihood-derivative diagnostic on the synthetic
and mouse datasets found the ordinary float64 fits within approximately
``5e-8`` in h2 of the reference optima, with likelihood loss below ``4e-13``.
The largest relative P-value difference from those reference fits was about
``3.1e-7`` (``0.000031%``). This explains the independent-fit roundoff without
changing the optimizer or a test tolerance; it is an observed example, not a
universal error bound.

.. warning::

   GPU float32 results can differ more from CPU float64 than CPU float32 results
   do. For the mouse dataset (1,940 samples and 10,150 SNPs), the largest observed
   difference was approximately 0.0004 in log10(p), or 0.09% relative P-value
   error. With the same fitted h2, SNP ``rs13479446`` had P-value
   ``2.0918834e-87`` on CPU float64 and ``2.0899597e-87`` on GPU float32.
   Both are extremely small, but their relative difference is about 0.092%.

   This example is an observation, not an upper bound for other datasets or
   hardware. Check numerical differences on representative data, including
   associations near a decision threshold. Use float64 when closer agreement
   is required. Float64 statistics computed after a float32 decomposition
   cannot recover precision already lost in that decomposition.

The same array dtype does not guarantee identical solver results. NumPy and
CuPy both receive float32 matrices and return float32 eigenvalues and
eigenvectors, but their LAPACK and cuSOLVER implementations and reduction orders
can round differently. For the 1,940-sample mouse example, CuPy float32 differed
from NumPy float64 by up to approximately 0.0004 in log10(p), even with h2 held
fixed. A diagnostic that explicitly promoted the same GPU float32 kernel to a
double-precision solve reduced the errors on the two worst chromosomes to about
0.00001. These are measured examples, not universal error bounds. Float64 is
needed when the observed float32 errors exceed a study's accuracy requirement;
check all association results, including very small p-values and near-null ties.
Agreement of the strongest associations alone is insufficient.

Epistasis requires CPU float64 computation. If ``ARRAY_MODULE`` or
``ARRAY_DTYPE`` requests another configuration, ``epistasis`` emits a
``UserWarning`` naming the requested settings and the actual CPU float64
configuration, then restores the caller's settings on return or failure.
Reader operations in workers explicitly use NumPy float64 as well. Float32 input
arrays are widened, but this cannot undo their original quantization. An
existing float32 eigen cache is rejected: remove that cache or use a new cache
path to recompute it in float64. Native float32 epistasis is unqualified; its
legacy fixture failures were preserved as evidence rather than weakening tests.

Choose precision and memory use
-------------------------------

Float32 stores the main LMM covariates, phenotypes, kernel/factor, decomposition,
and test SNP arrays in single precision on both CPU and GPU. Kernel-diagonal
normalization, h2 likelihood calculations, and per-SNP statistics deliberately
use float64 on both backends, but cannot recover precision already lost in the
decomposition or rotations.
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

The test-SNP path derives the variance for ``EffectSize`` from Unit training
statistics, avoiding a second genotype read. It squares the trained standard
deviation in float64 and maps the constant-SNP infinity sentinel to zero;
float32 training moments still carry float32 rounding.

BED decoding runs on the CPU through bed-reader. PySnpTools transfers values
with ``cupy.asarray`` and standardizes them with its CuPy kernels before
FaST-LMM's GPU linear algebra. No bed-reader GPU build is required.
``pysnptools.util.gpu.has_gpu_support()`` checks CuPy and its current device.

The following example requires CuPy and a usable CUDA device::

    import cupy as cp
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from bed_reader import to_bed
    from pysnptools.util.gpu import has_gpu_support
    from fastlmm.association import single_snp
    from pysnptools.snpreader import Bed, SnpData

    if not has_gpu_support():
        raise RuntimeError("this example requires CuPy and a usable CUDA device")
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
are serialized between FaST-LMM workflow scopes within one process. An omitted ``xp``
resolves after preceding calls restore their environment. The lock does not
coordinate unrelated code changing ``os.environ``; explicit ``xp`` keeps the
workflow choice clear. Standalone reader behavior is described in
PySnpTools's GPU guide.

Two-kernel analyses (``K1`` or ``G1``) run on NumPy/float64. A requested GPU
backend or float32 precision produces a warning identifying that actual fallback.
Other numerical workflows use the same conservative boundary; see
:doc:`api_backend_precision` for the full API checklist, class methods, utilities,
worker behavior, and limitations.

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

These APIs are included in the published alpha. Its dependency bounds select
the coordinated PySnpTools and bed-reader alphas. Installed CPU/GPU workflows
were validated against all three published packages.


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
    from bed_reader import to_bed
    from pysnptools.util.gpu import has_gpu_support
    from fastlmm.association import single_snp
    from pysnptools.snpreader import Bed, SnpData
    from pysnptools.util import get_array_module
    from pysnptools.util.mapreduce1 import map_reduce
    from pysnptools.util.mapreduce1.runner import LocalMultiProc

    if not has_gpu_support():
        raise RuntimeError("this example requires CuPy and a usable CUDA device")

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

CPU read buffers, host-to-device copies, CuPy pools, temporary products,
the factor/kernel, and solver workspace all contribute to memory use; CUDA allocation failures propagate. Python HIP is not implemented.
CPU kernel formation uses the selected dtype, but reader, BLAS, and solver
workspace still contribute intermediate memory. No timing or memory guarantee
follows from ``dtype`` alone.
