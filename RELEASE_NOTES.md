# FaST-LMM release notes

## 1.0.0 (unreleased)

- Require Python 3.11.5 or later and prepare core Python 3.15 support while
  retaining Python 3.11 through 3.14. Final acceptance requires the compatible
  bed-reader, PySnpTools, and numerical-backend releases to be published first.
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
