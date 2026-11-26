# FaST-LMM release notes

## 1.1.0 (in beta; changes since 1.0.0)

Highlights:

- **GPU and float32 for single-kernel `single_snp`.** With `xp="cupy", dtype="float32"`,
  it ran over 10 times as fast on an NVIDIA RTX 3060 as the default CPU float64, and
  about 5 times as fast as CPU float32, on a synthetic dataset with 14,925 individuals
  and 100,000 SNPs. Install with `pip install "fastlmm[cupy]"`. A 12 GB card handles
  about 20,000 individuals with a full-rank kernel. See the GPU guide.
- **Faster CPU float32.** CPU float32 `single_snp` builds its kernel in float32 and runs
  its eigendecomposition and low-rank SVD with SciPy's float32 LAPACK drivers (NumPy's
  `linalg` computes float32 inputs in float64). At the same thread count it ran about
  2.5 times as fast as CPU float64. Float32 p-values differ slightly from float64 (on
  the synthetic dataset, up to 1.7e-3 in log10 P, with the same top SNPs). Float64
  results are unchanged.

Other changes:

- `single_snp` takes `xp` (`"numpy"` or `"cupy"`) and `dtype` (`"float32"` or
  `"float64"`). When omitted, the `ARRAY_MODULE` and `ARRAY_DTYPE` environment
  variables supply the defaults, then NumPy and float64. Workers inherit the settings,
  and `single_snp` restores the caller's environment afterward.
- Every other numerical function, including two-kernel `single_snp`, epistasis, and the
  predictors, warns and runs on the CPU in float64 if a GPU or float32 is requested. A
  cached model holding float32 or CuPy arrays raises `ValueError`; refit it in float64.
- GPU float64 emits PySnpTools' `GpuFloat64Warning`, because float64 can be much slower
  than float32 on consumer GPUs. With float32, covariate and phenotype values too large
  or too small for float32 raise `ValueError`.
- Fix `cache_file` with a supplied `h2` or `mixing`, and multiple phenotypes with a
  supplied `h2`. A supplied `h2` now takes precedence over a cached one, and a supplied
  `mixing` that disagrees with the cache raises `ValueError`. Caches also record their
  dtype, and a mismatch raises `ValueError`. Caches from 1.0 still work with float64.
- Remove the unused `standardize_unit_python` and `standardize_beta_python` from
  `fastlmm.util.standardizer`; they predate PySnpTools and standardized constant
  fractional SNPs to +/-1. Use PySnpTools' `Unit` and `Beta` standardizers.
- Fix a crash in heritability and mixing searches on nearly flat likelihoods, where
  SciPy's `brent` rejected a grid bracket with `ValueError`. Results are unchanged
  wherever the search succeeded before.
- Stop bundling the legacy FaST-LMM C++ program (`fastlmmc`), which shrinks the wheel
  by about 40 MB. Its only use was `FastLmmSet(autoselect=True)`, an undocumented
  option that `snp_set` never used; `autoselect` now defaults to `False`, and `True`
  raises `ValueError` (pass the chosen SNPs as `filenull` instead).
  `fastlmm.util.run_fastlmmc` is removed. FaST-LMM's tests still compare against
  `fastlmmc` 2.08 (Apache License 2.0) from the repository's `external/fastlmmc`.
- Require PySnpTools 1.1 and bed-reader 1.3 (during the beta, 1.1.0b2 and 1.3.0b3 or
  later). bed-reader now reads on the CPU only; PySnpTools copies the data to the GPU.
  The new `cupy` extra installs GPU support through PySnpTools.
- GPU support targets NVIDIA CUDA on x86-64 Windows and Linux, including WSL2. Python
  3.15 is CPU-only until CuPy supports it.
- New GPU guide, README section, and a float32 and GPU section in the main notebook. The
  notebook's deprecated LMM(select)+PCs section is replaced by a link to its last
  executed version.

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
