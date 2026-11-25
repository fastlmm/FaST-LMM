# FaST-LMM release notes

## Unreleased

- **GPU and float32 for single-kernel `single_snp`.** With `xp="cupy", dtype="float32"`,
  it ran about 10 times as fast on an NVIDIA RTX 3060 as the default CPU float64, and
  about 5 times as fast as CPU float32, on a synthetic dataset with 14,925 individuals
  and 100,000 SNPs. Install with `pip install "fastlmm[cupy]"`. Other functions warn and
  run on the CPU in float64. See the GPU guide.
- **Faster CPU float32.** CPU float32 `single_snp` now builds its kernel in float32 and
  runs its eigendecomposition and low-rank SVD with SciPy's float32 LAPACK drivers
  (NumPy's `linalg` computes float32 inputs in float64). At the same thread count it
  ran about 2.5 times as fast as CPU float64. CPU float32 p-values therefore differ
  slightly from earlier releases (on the synthetic dataset, up to 1.7e-3 in log10 P
  from CPU float64, with the same top SNPs). Float64 results are unchanged.
- Use CPU-only bed-reader: test SNPs from any reader, including BED, are decoded
  on the CPU, then copied to CuPy for GPU standardization and computation.
- PySnpTools provides `GpuFloat64Warning` and `has_gpu_support` in
  `pysnptools.util.gpu`.
- Require bed-reader 1.3.0b3 and PySnpTools 1.1.0b2 or later. The `cupy` extra
  installs GPU support through PySnpTools.
- Recheck small float32 residuals in float64 to preserve signal that would
  previously have been zeroed. Rebuild version 3 float32 caches before reuse.
- Preserve NumPy/float64 execution in same-process threaded workers without
  deadlocking the parent workflow.
- Fix a crash in heritability and mixing searches on nearly flat likelihoods. When
  re-evaluating a grid triplet gave slightly different values, SciPy's `brent`
  rejected the bracket with `ValueError`; FaST-LMM now searches that interval with a
  bounded search instead. Results are unchanged wherever the search succeeded before.
- Bundle FaST-LMM C++ (`fastlmmc`) 2.08 in place of 2.07, for the deprecated
  `compute_auto_pcs`/`single_snp_select` path. 2.08 is licensed under the
  Apache License 2.0 (2.07 was non-commercial only), builds with oneMKL
  2026.2.1, and runs on current Intel and AMD CPUs; the 2.07 Linux binary
  crashed with "Illegal instruction" on current CPUs. It computes the same
  results and ran 1.6 to 2.4 times as fast in our tests. Its license, notice,
  and third-party licenses are installed beside the executables. The launchers
  also no longer set `FastLmmUseAnyMklLib` in the caller's own environment,
  only in the child process.

## 1.1.0a1 (2026-11-18, alpha)

- Add the `cupy` extra and require the coordinated PySnpTools 1.1.0a1 and
  bed-reader 1.3.0a1 alphas. CPU execution remains the default.
- Add `single_snp` NumPy/CuPy selection and float32/float64 workflow precision
  through explicit arguments or `ARRAY_MODULE` and `ARRAY_DTYPE`. Propagate
  defaults to workers and restore the environment on success or failure.
- Preserve CubeCL-backed GPU SNP storage where supported; fix chromosome
  covariate alignment and validate raw values before narrowing or transfer.
- Report consumer-GPU float64 arithmetic costs without changing precision.
  Retain higher precision for internal reductions and likelihood statistics.
- Initial GPU scope is NVIDIA Windows/Linux x86-64, including WSL2, on Python
  3.11–3.14. Python 3.15 remains CPU-only pending CuPy support. Published-alpha
  integration tests passed on native Ubuntu 24.04 with an RTX 3060 and Python
  3.11, in addition to Windows and WSL2 validation.
- All three coordinated alphas are published on PyPI. Dependencies resolve
  against the published PySnpTools and bed-reader alphas; the development-only
  Git source has been removed.

## 1.0.0 (2026-11-14)

- Require Python 3.11.5 or later and support Python 3.11 through 3.15.
- Make HDF5 support optional. Existing HDF5 users must install
  `fastlmm[hdf5]`; install `fastlmm[hdf5,bgen]` for both optional formats.
  HDF5 and BGEN remain qualified on Python 3.11 through 3.14 while their
  Python 3.15 binary dependencies are unavailable.
- Refresh stable runtime and development dependencies, resolving the locked
  Pillow and Soup Sieve security findings. Declare bed-reader and tabulate as
  direct runtime dependencies.
- Expand installed-artifact checks and stable-interpreter CI qualification;
  retain the canonical integration suite rather than substituting incomplete
  pytest collection.
- Require bed-reader 1.2.0, PySnpTools 1.0.0, and fastlmmrslib 0.1.0 or later.
  PySnpTools 1.0 no longer needs `six`, so FaST-LMM no longer declares it.
- Support Windows ARM64 on Python 3.13 and later. Python 3.11 and 3.12 are not
  supported there because statsmodels publishes Windows ARM64 wheels only for
  Python 3.13 and later.
- Replace fastlmmclib with fastlmmrslib, a Rust implementation of Davies'
  algorithm, for SNP-set score-test p-values. A Davies fault now raises
  `ArithmeticError` instead of returning an out-of-range p-value.
- Refresh the saved outputs of the maintained notebooks, and make the
  SingleSnpScale examples use a temporary folder instead of `c:\deldir`.
- Deploy strict Sphinx builds through GitHub Pages artifacts. Remove tracked
  generated HTML after verifying the deployed pages and static assets.
- Require NumPy 2.3.5 or newer and SciPy 1.16.1 or newer on Python 3.14 while
  retaining compatible lower bounds on older Python versions.
- Support pandas 3 while excluding the next untested major release with
  `pandas<4`.
- Correct NumPy scalar conversions and the Matplotlib legend-handle access for
  current dependency releases.
- Migrate packaging to `uv_build`, add reproducible locked and lower-bound
  testing, and add isolated wheel and source-distribution checks.
- Add a tag-triggered PyPI Trusted Publishing workflow. Publication still
  requires explicit maintainer approval through the protected `pypi`
  environment.
