# FaST-LMM release notes

## 0.6.13 (unreleased)

- Add and test support for Python 3.14 while retaining Python 3.10 through
  3.13.
- Require the Python 3.14-compatible prerequisite releases
  `pysnptools>=0.5.15` and `fastlmmclib>=0.0.8`.
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
