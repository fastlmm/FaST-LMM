GPU association and floating-point precision
============================================

Single-kernel :func:`.single_snp` can run on an NVIDIA GPU, in float32 (strongly
recommended) or float64, thanks to `CuPy <https://cupy.dev/>`_, the NumPy-compatible GPU array
library. NumPy on the CPU remains the default.

**Over 10× faster with GPU and float32.** On a synthetic dataset with 14,925
individuals and 100,000 SNPs, single-kernel ``single_snp`` ran over 10 times as
fast with float32 on an RTX 3060 as with the default CPU float64, and about 5 times
as fast as CPU float32. On the CPU alone, float32 ran about 2.5 times as fast as
float64. The CPU runs used 16 threads on a Ryzen 9 7950X. Most of the time goes to
the eigendecomposition and the association tests, which run on the GPU. See the
:doc:`benchmark <benchmark>` for details and how to reproduce it.

We recommend float32 for GPU work. Consumer GPUs like the RTX 3060 process float32
arithmetic much faster than float64. Float32 changes p-values slightly, so compare against float64 on representative
data when precision matters.

Quick start
-----------

This complete example uses FaST-LMM's example data, which it downloads on first use:

.. code-block:: python

    from fastlmm.association import single_snp
    from fastlmm.util import example_file

    bed_fn = example_file("tests/datasets/synth/all.*", "*.bed")
    pheno_fn = example_file("tests/datasets/synth/pheno_10_causals.txt")
    cov_fn = example_file("tests/datasets/synth/cov.txt")

    results = single_snp(bed_fn, pheno_fn, covar=cov_fn, count_A1=False,
                         xp="cupy", dtype="float32")
    print(results.head())

The result is an ordinary pandas DataFrame on the CPU (see `Results`_). To see what
float32 changes, run the same call with ``dtype="float64"`` and compare p-values;
to separate precision from the GPU, also run ``xp="numpy", dtype="float32"``.

Installing GPU support
----------------------

Install FaST-LMM with the ``cupy`` extra::

    pip install "fastlmm[cupy]==1.1.0b1"

From a source checkout, run ``uv pip install ".[cupy]"`` in a fresh virtual
environment instead. FaST-LMM requires PySnpTools 1.1.0b2 or later; pip installs it
automatically.

GPU support requires NVIDIA CUDA on x86-64 Windows or Linux, including WSL2. Linux
GPU wheels require glibc 2.35 or later. Python 3.15 is currently CPU-only because
CuPy has no Python 3.15 wheel yet. See
`CuPy's installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for
driver and CUDA setup. Omit ``[cupy]`` for CPU-only use.

``pysnptools.util.gpu.has_gpu_support()`` reports whether CuPy can use the current
device. Requesting ``xp="cupy"`` without CuPy installed raises ``ImportError``, and
GPU errors propagate to the caller.

Genotypes are always decoded on the CPU (bed-reader needs no GPU build), then
PySnpTools copies them to the GPU and standardizes them there. See
`PySnpTools's GPU guide <https://fastlmm.github.io/PySnpTools/gpu.html>`_ for
reading and standardizing data directly.

.. _choosing-precision:

Choosing precision
------------------

With ``dtype="float32"``, the covariates, phenotypes, kernel, decomposition, and
test SNPs are stored in single precision on both the CPU and the GPU. The h2
likelihood and the per-SNP test statistics are computed in float64, but that cannot
recover precision already lost in the float32 decomposition.

.. warning::

   Float32 results differ from float64, and GPU float32 can differ more than CPU
   float32 because GPU and CPU solvers round differently. On the mouse dataset
   (1,940 individuals), the largest difference was about 0.0004 in log10(p), about 0.09%
   relative error in the p-value. That is one observed example, not a bound for
   other data. Check representative results, including associations near your
   significance threshold, and use float64 when you need closer agreement.

Float64 works on GPUs, but it can be much slower than float32 on consumer GPUs
(data-center GPUs such as the A100 and H100 have fast float64). When you use float64
on a GPU, PySnpTools warns about this with ``GpuFloat64Warning``. The warning changes
nothing. If you want float64 anyway, you can silence it:

.. code-block:: python

    import warnings
    from pysnptools.util.gpu import GpuFloat64Warning

    warnings.filterwarnings("ignore", category=GpuFloat64Warning)

With float32, covariate or phenotype values too large or too small for float32 to
handle safely raise ``ValueError``; rescale them or use float64.

What runs on the GPU
--------------------

Currently, only single-kernel ``single_snp`` runs on the GPU or in float32. Please
`open an issue on GitHub <https://github.com/fastlmm/FaST-LMM/issues>`_ if you would
like GPU support for another function. Two-kernel
analyses (``K1`` or ``G1``), epistasis, the predictors, and the other numerical
functions warn and run on the CPU in float64, whatever ``xp``, ``dtype``, or the
environment requests. :doc:`api_backend_precision` lists every function.

Epistasis rejects an existing float32 eigen cache; remove it or use a new cache path.

Backend and precision defaults
------------------------------

For a single call, pass ``xp=`` and ``dtype=``; they override everything else. To
set defaults for a whole script or notebook, set the environment variables
``ARRAY_MODULE`` (``numpy`` or ``cupy``) and ``ARRAY_DTYPE`` (``float32`` or
``float64``). Without either, FaST-LMM uses NumPy and float64. To set them for just
a block of code, use the standard library's ``unittest.mock.patch.dict``, which
restores the previous settings when the block ends, even after an exception:

.. code-block:: python

    from unittest.mock import patch

    with patch.dict("os.environ", {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}):
        results = single_snp(bed_fn, pheno_fn, covar=cov_fn, count_A1=False)

These settings are process-wide, so other threads see them. ``single_snp`` sets
them for its own run and restores them afterward, and overlapping ``single_snp``
calls in one process run one at a time.

With a ``runner``, each worker uses its own environment, including runner
overrides, and model arrays move to that worker's backend. So a CuPy call in the
parent does not by itself put every worker on the GPU; configure worker
environments consistently. A worker ``ARRAY_DTYPE`` that disagrees with the
workflow's precision raises ``ValueError``.

Reusing precomputation
----------------------

``cache_file`` saves the decomposition so later calls with the same phenotype,
covariates, and kernel inputs can skip it; only the test SNPs may differ. It is not
keyed by input contents, so use separate paths for separate analyses and
precisions. A cache made with the other dtype raises ``ValueError``. Rebuild float32
caches made by FaST-LMM 1.1.0a1; float64 caches from earlier releases still work.

A supplied ``h2`` takes precedence over a cached one, so the same cache can serve
several heritabilities:

.. code-block:: python

    first = single_snp(bed_fn, pheno_fn, count_A1=False, h2=0.5, cache_file="study.npz")
    second = single_snp(bed_fn, pheno_fn, count_A1=False, h2=0.6, cache_file="study.npz")

If a cache was made with a supplied ``h2`` and a later call omits it, the later call
fits ``h2``. An explicit ``mixing`` that disagrees with the cached mixing raises
``ValueError``. With leave-one-chromosome-out (the default), each chromosome gets
its own cache file, and multiple phenotypes add a ``.extra.npz`` file.

.. _gpu-memory:

Memory
------

A float32 run needs about 28 bytes × individuals × the smaller of individuals and
kernel SNPs, plus about 1 GB. On a 12 GB card, that is about 20,000 individuals with
a full-rank kernel, or, with a low-rank kernel, about 50,000 individuals with 8,000
kernel SNPs. Larger runs fail on Linux; on Windows and WSL2 they borrow system memory
and run much more slowly.

The dense kernel is individuals by individuals, and the decomposition, CuPy's memory pool,
and temporary products need GPU memory too. ``GB_goal`` sets how many test SNPs are
processed at once; it is a target, not a cap on GPU memory, and it can raise the
target (with a warning) to keep blocks large enough. Low-rank analysis still needs
the whole genotype factor in memory. CUDA allocation errors propagate; there is no
automatic CPU retry.

Results
-------

``single_snp`` returns a pandas DataFrame on the CPU, sorted by ``PValue``, with
columns ``SNP``, ``sid_index``, ``Chr``, ``GenDist``, ``ChrPos``, ``PValue``,
``SnpWeight``, ``SnpWeightSE``, ``EffectSize``, ``Mixing``, and ``Nullh2``
(``show_snp_fract_var_exp=True`` adds ``SnpFractVarExpl``). With multiple
phenotypes there is one row per SNP and phenotype, with ``Pheno`` and
``PhenoCount`` columns. ``output_file_name`` also writes the results as
tab-delimited text. See :func:`.single_snp` for every option.
