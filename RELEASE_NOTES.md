# FaST-LMM release notes

## Unreleased

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
- Run the bundled legacy `fastlmmc` executable on AMD CPUs: FaST-LMM's
  launchers now set `MKL_CBWR=COMPATIBLE` for the child process unless
  `MKL_CBWR` or `MKL_DEBUG_CPU_TYPE` is already set. They also no longer set
  `FastLmmUseAnyMklLib` in the caller's own environment. Direct shell calls to
  `fastlmmc` on AMD still need `MKL_CBWR=COMPATIBLE`.

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
