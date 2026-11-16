GPU association and floating-point precision
============================================

These features are under development on the GPU branches; installing the latest
published package does not necessarily provide them. The examples below target
this branch's API.

Run a small association study
-----------------------------

``single_snp`` accepts ``xp="numpy"`` or ``xp="cupy"`` and
``dtype="float64"`` (the default) or ``dtype="float32"``. NumPy runs on the CPU;
CuPy requires a working GPU installation. Import CuPy explicitly and check the
device before requesting it: if CuPy cannot be imported, backend resolution can
log a warning and fall back to NumPy. A successful import alone does not verify
that the driver can execute GPU operations.

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

Float32 stores the main covariates, phenotypes, kernel, decomposition, and test
SNP arrays in single precision. Per-SNP statistics use float64 arithmetic, but
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

With a GPU-enabled bed-reader build (the ``gpu-cuda`` build feature) and CuPy,
this branch can read and Unit-standardize BED blocks on the GPU and form the
kernel or low-rank factor there. The test-SNP path also obtains the raw variance
needed for ``EffectSize`` from the GPU block, avoiding a second CPU read.
Selecting CuPy alone does not enable native GPU BED decoding. In the committed
implementation, an unavailable native GPU reader uses the existing reader path;
other input formats and custom standardizers can also use that path.

For your own BED/phenotype files, the complete call is::

    from fastlmm.association import single_snp
    from pysnptools.snpreader import Bed, Pheno

    results = single_snp(
        test_snps=Bed("cohort.bed", count_A1=True),
        pheno=Pheno("cohort.phe"),
        xp="cupy",
        dtype="float32",
        GB_goal=2,
    )

``cohort.bim`` and ``cohort.fam`` must accompany the BED file. Keep
``count_A1`` explicit so allele coding is clear. LOCO remains enabled by default;
using ``leave_out_one_chrom=False`` can cause proximal contamination.

Backend scope and limitations
-----------------------------

An explicit ``xp`` overrides ``ARRAY_MODULE``; otherwise the environment selects
the backend, with NumPy as the final default. ``single_snp`` temporarily sets
``ARRAY_MODULE`` for the workflow and restores it afterward. Treat this as a
process-wide setting when arranging concurrent workflows and runners; configure
worker environments consistently. Standalone reader behavior is described in
PySnpTools's GPU guide.

Two-kernel analyses (``K1`` or ``G1``) run on NumPy. A requested GPU backend
produces a warning in this case. This guide covers ``single_snp``; it does not
promise GPU or float32 support for every FaST-LMM entry point.

Reuse precomputation safely
---------------------------

A cache requires the same phenotype, covariates, and training inputs; it is not
a general cache keyed by input contents. Use distinct paths for distinct
analyses and precisions. A cache with another dtype raises ``ValueError``.
Using the in-memory data from the first example::

    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        cache = str(Path(directory) / "cpu64.npz")
        first = single_snp(snps, phenotype, xp="numpy", dtype="float64",
                           h2=0.5, cache_file=cache)
        second = single_snp(snps, phenotype, xp="numpy", dtype="float64",
                            h2=0.6, cache_file=cache)
        assert len(first) == len(second) == len(sid)

An explicitly supplied ``h2`` takes precedence when reusing the decomposition.
If a cache was created with a supplied ``h2`` and a later call omits it, the later
call fits ``h2``. Multiple phenotypes can use a supplied ``h2`` too. Changing an
explicit ``mixing`` to disagree with the cached mixing raises ``ValueError``;
create a separate cache for that model. LOCO creates a separate cache per
chromosome, and multiple phenotypes can create an additional ``.extra.npz`` file.

Coordinated reader API preview
------------------------------

Concurrent implementation work moves BED GPU reads and standardization into
PySnpTools's reader API. The ``single_snp(..., xp=..., dtype=...)`` calls above
remain the public FaST-LMM workflow; users do not need private native-reader
helpers. The PySnpTools API preview documents explicit reader ``xp`` arguments
and GPU kernel construction for standalone use.

In that implementation, requesting BED reads on the GPU requires a matching
GPU-enabled bed-reader build; a missing native build raises. This replaces the
committed native-reader fallback described above. Existing arrays determine the
device for standalone standardization, while explicit reader ``xp`` arguments
override the environment at each read boundary.

``single_snp`` establishes the workflow's ``ARRAY_MODULE`` and restores it on
success or failure. Overlapping calls within one process are serialized to
protect that process-wide context. Runner workers resolve their backend from
their own environment, including explicit runner overrides; model arrays move to
the worker's backend. A parent CuPy call alone therefore does not prove every
worker ran on the GPU. Two-kernel analyses continue to use NumPy.

These coordinated changes are present in the original development checkout,
not this documentation worktree's committed implementation. Their GPU builds,
worker behavior, and published dependency bounds require coordinated acceptance
before release.
