# gx2 1.6 and 1.6.1 reevaluation against fastlmmrslib

<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

Assessment date: November 15, 2026. Worktree branch: `evaluate-gx2`, based on
freshly fetched `origin/main` at `a40f1735d68b8982d7f57d55d843c21048ba5411`.
This supplements the historical [gx2 1.5 evaluation](QUADRATIC_FORM_EVALUATION.md).

## Implementation and qualification

The candidate on `evaluate-gx2` now replaces FaST-LMM's runtime dependency on
fastlmmrslib with `gx2==1.6.1`. The exact pin keeps upgrades subject to
deliberate qualification. This section describes the implementation;
the evaluation sections below retain the evidence preceding that change.
The separately published fastlmmrslib project and bundled legacy executables
are not retired by this change.

The private adapter is
[`_quadratic_form.py`](../fastlmm/association/_quadratic_form.py), called through
the existing `Sc.pv_davies_eig` entry point. Public SNP-set test names and
signatures are unchanged. The adapter:

- Validates finite real scalar/vector inputs and finite scalar probabilities
  in `[0, 1]`, retaining negative roundoff coefficients.
- Preserves the old all-zero/empty-spectrum result of one at statistic zero,
  with exact support checks for nondegenerate same-sign spectra.
- Uses gx2 automatic dispatch with `AbsTol=1e-14, RelTol=1e-12`; the absolute
  acceptance target remains `1e-7`.
- Distinguishes boolean integration flags from numeric Ruben error bounds,
  rejecting nonfinite, malformed, negative, or excessive diagnostics.
- Accepts a flagged upper tail only when the fixed-parameter Chernoff bound
  independently verifies the existing absolute-accuracy target and the
  reported probability is inside that bound. Comparisons use logarithms to
  prevent bound underflow from masquerading as an exact zero.
- Raises `ArithmeticError` for unverified results, preserves backend
  exceptions, leaves numerical warnings visible, and uses no random sampling.

The lockfile changes only the removed Rust package and added gx2/mpmath
packages, with corresponding dependency metadata. Existing CRLF line endings
are preserved. The regression suite covers independent positive/signed tails,
legacy point-mass behavior, scale invariance, bound acceptance/rejection,
invalid inputs/outputs/diagnostics, exception preservation, negative roundoff
coefficients, and ordinary-path threaded repeatability. Installed-artifact
smoke tests now exercise the score-test adapter and check dependency metadata.

### Validation results

The canonical `tests/test.py` runner is unchanged. Independent cached source
snapshots prevent its legacy output files from colliding across environments.
Their runtime code was checked against the candidate, and all environments
selected gx2 1.6.1. The final exact dependency pin changes metadata only;
Rebuilt artifacts were rechecked separately. All nine full runs below passed,
with 193 canonical tests each, including the new adapter tests.
Core-only runs retain five preexisting HDF5-dependent skips; no test was
newly skipped or deselected.

| Linux x86-64 environment | Canonical suite |
| --- | --- |
| Python 3.11, locked, all extras | Passed, 193 tests |
| Python 3.12, locked, all extras | Passed, 193 tests |
| Python 3.13, locked, all extras | Passed, 193 tests |
| Python 3.14, locked, all extras | Passed, 193 tests |
| Python 3.15, core published binary dependencies | Passed, 193 tests, 5 existing HDF5 skips |
| Python 3.11, lowest direct dependencies, HDF5/BGEN | Passed, 193 tests |
| Python 3.14, lowest direct dependencies, HDF5/BGEN | Passed, 193 tests |
| Python 3.11, lowest direct dependencies, core only | Passed, 193 tests, 5 existing HDF5 skips |
| Python 3.14, lowest direct dependencies, core only | Passed, 193 tests, 5 existing HDF5 skips |

Both wheel and sdist were built with `uv build --no-sources`, and a wheel was
also rebuilt from the sdist. Archive checks verify the new module, metadata,
license, authors file, sample data, and legacy executable, and reject generated
test outputs in the wheel. The four clean Linux artifact environments
(Python 3.11 and 3.15, wheel and sdist) pass smoke tests plus all 12 adapter
regression tests. A clean installed wheel works with fastlmmrslib absent.

Windows x86-64 Python 3.13 independently passes both wheel and sdist smoke
tests plus all 12 adapter regression tests for each artifact. All six score
integration cases also pass against the unchanged Windows-specific reference
outputs, including reordered-ID and two-kernel cases.

Repository-wide Ruff using CI's pinned 0.17.5 passes, `uv lock --check` passes,
and Sphinx builds cleanly with warnings treated as errors. No numerical
reference outputs or existing test tolerances were changed. The executable
mode changed by legacy testing was restored after the run.

Exact dependency freezes and `qualification-results.json` retain all nine
canonical results and candidate artifact hashes. Detailed canonical, artifact,
documentation, and Windows integration logs are in `/mnt/c/deldir/gx2-evaluation`; reproducible qualification drivers are in
that same findable folder. Linux environments and isolated source snapshots
are under `/home/carlk/.cache/gx2-evaluation`; Windows environments are under
`C:\deldir\gx2-evaluation`. All prerequisite packages came from published
artifacts; no sibling library implementation is used for qualification.

The full macOS/Windows/ARM CI matrix remains required before release. macOS
and ARM runtimes are unavailable locally; local Windows coverage is explicitly
limited to Python 3.13 artifact and score integration checks. Release notebook
execution was not performed for this backend-only change. The package version
remains 1.0.0 for these local candidate builds; the maintainer must choose a
new version before publishing. No package release or release tag was published.

## Evaluation before implementation: gx2 1.6.1

Tested the published [gx2 1.6.1](https://pypi.org/project/gx2/1.6.1/) wheel
on November 15, 2026, after the
[maintainer reported both follow-up fixes](https://github.com/abhranildas/gx2-py/issues/1#issuecomment-6898407064).
Wheel SHA-256:
`c137ebe1dc7252f5cf645bb09dd69dddcb503cc254016aff3cb7e02fa3c64190`.
Separate cached environments preserve the previous 1.6 installations and use
the same dependency versions for a controlled comparison.

**Both reported defects are fixed. Moving forward with a replacement adapter
is now reasonable under FaST-LMM's existing absolute-accuracy contract.**
The unmodified `auto` route still flags three real tails; an independently
bounded acceptance policy resolves those cases in the integration experiment.
This is evidence supporting implementation, not full release qualification.
The earlier recommendation below is retained as the historical 1.6 assessment.

### Fix verification and numerical results

- Both Python 3.11 and 3.15 pass all 45 independent-reference cases in the
  71-distribution tight-tolerance screen, including the formerly NaN boundary.
  Python 3.15 also passes all 45 with default settings.
- Both environments pass 288 boundary checks each without warnings. These
  cover all eight selectable methods, scalar and array CDF/SF values below,
  at, and above a point mass, nonzero offsets, all-zero weights, all-zero
  degrees of freedom/noncentralities, and the documented PDF convention.
- The original four-dimensional `ray` reproducer returns approximately
  `2.06321684e-9`, matching the elementary reference
  `2.063216839277836e-9`. It takes about 3.8 seconds at default tolerances
  on Python 3.15. Python 3.11 also passes three-dimensional chi-square and
  four-dimensional unequal-weight ray checks.
- All 60 analytic sweep calls complete without exceptions and meet the
  `1e-7` absolute criterion. All six formerly broken ray calls now return
  accurate nonzero results, including tails near `1.93e-22`. With tight
  tolerances these take approximately 3.8–17.2 seconds each. The per-node
  ray-underflow warning repeats many times when warnings are recorded with
  the `always` filter; warning counts and raw messages are retained in the
  JSONL. These calls do not provide a quadrature error estimate.
- The serial/four-thread ordinary-path repeatability checks still pass on
  both Python versions.

### The three flagged real tails

Strict `auto` integration still produces three errors in the six-test suite;
1.6.1 does not claim to remove the underlying Imhof cancellation limitation.
The flags and spectra are unchanged, including tiny negative eigenvalues.

The repaired 47-dimensional `ray` route runs without warnings on these
inputs. However, it uses Monte Carlo above four dimensions. For one spectrum,
three deterministic seeds with 50,000 rays give approximately `8.16e-20`,
`2.40e-22`, and `2.83e-22`, with standard-error estimates comparable to or
larger than the estimates. The default 500-ray result is about `8.56e-24`.
Consequently, a working ray method alone does not establish tail-relative
accuracy or justify interpreting these estimates as precise tiny p-values.
No random eigenvalue truncation, sign correction, or spectrum modification
was performed.

An independent bound is sufficient for the current absolute contract. For
`Q = sum(w_i * Z_i**2)` and any `0 < t < 1/(2*max(w_i))`, Markov's inequality
and the chi-square moment-generating function give

```text
P(Q > x) <= exp(-t*x - 0.5*sum(log(1 - 2*t*w_i))).
```

This remains valid with negative coefficients. Choosing the fixed interior
`t = 0.9/(2*max(w_i))` gives an upper bound of about `1.7714e-17` for each of
the three captured tails. Numerically optimizing t tightens the bound to
about `2.1056e-18`, but the experiment does not depend on that optimization.
These are analytic probability bounds evaluated in double precision, not
certified interval-arithmetic enclosures; the margin below `1e-7` is more
than nine orders of magnitude.

A separate process-local integration experiment retains every original flag
and accepts a flagged returned probability only if
`0 <= probability <= independent_upper_bound <= 1e-7`. All other invalid
probabilities and unacceptable diagnostics still raise. **All six integration
tests pass with this policy**, covering 138 calls and the three flagged zeros.
For these specific calls, zero meets the established absolute accuracy
requirement; this does not claim the true probability is zero or provide
relative accuracy. It also does not qualify all unflagged gx2 results beyond
the numerical screening performed above.

### Recommendation and remaining work

Proceed toward a focused gx2 adapter if eliminating native-package maintenance
is the goal. The two upstream defects no longer block that work, and the
three flagged examples now have an independently justified acceptance route.
Performance parity with Rust is not an adoption requirement for this workflow.

The production adapter still needs an explicit policy for arbitrary flagged
inputs: accept only with a valid accuracy bound or a separately qualified
method, otherwise raise. It must also decide how to preserve the existing
Rust zero-spectrum boundary convention, which differs from gx2's strict
survival definition. No production code or dependency changes have been made.
Full FaST-LMM, installed wheel/sdist, other supported Python/OS/architecture,
minimum-dependency, and optional-feature qualification remains outstanding.

New drivers and results are in `/mnt/c/deldir/gx2-evaluation/v1.6.1`;
environments are `/home/carlk/.cache/gx2-evaluation/v161-py315` and
`v161-py311`. The new drivers are `check_boundaries.py`, `check_flagged.py`,
and `integration_bounded.py`. The historical numerical and analytic drivers
are reused from the parent directory. `flagged-rays.jsonl` records seeds,
standard-error estimates, and independent bounds; `integration-bounded.log`
and `bounded/gx2-real-inputs.jsonl` retain the passing integration evidence
without discarding the original gx2 failure flags. The strict integration
log retains the expected three failures. Published metadata and environment
versions are saved alongside these results. Legacy executable modes were
restored after testing; expected outputs were not modified.

## Historical recommendation: gx2 1.6

Keep `fastlmmrslib` as FaST-LMM's default backend for now. Published gx2 1.6
fixes the earlier decisive positive-weight accuracy failures and deserves
consideration again. It is technically possible to adapt FaST-LMM to use it,
but it is not a qualified direct replacement today:

- Three real integration calls return clipped zero probabilities with a true
  failure flag. Respecting the new diagnostics causes three of six existing
  integration tests to error. Ignoring them makes all six tests pass.
- Scalar performance is a measured tradeoff, not an established adoption
  blocker: all 138 calls total about 0.33 seconds with defaults or 0.65 seconds
  with tight settings. This calculation is used by the optional Davies SNP-set
  score-test path, not the ordinary single-SNP association path.
- Degenerate spectra still require explicit handling, and selecting a reliable
  alternative for flagged or very small tails requires further qualification.

The practical benefit would be a universal Python wheel and reliance on an
upstream numerical implementation, avoiding maintenance of our own native
quadratic-form package. If this calculation is outside the performance-critical
workload, that maintenance benefit can reasonably outweigh the measured scalar
cost. The large kernel ratios should not drive the adoption decision by
themselves. The remaining blockers are numerical-policy decisions and adapter
validation, especially a reliable treatment of the flagged tails. Once those
are resolved, switching is a reasonable direction; a representative workflow
timing can check the practical cost without requiring kernel-speed parity.

No production code, dependencies, lockfile, expected outputs, or tolerances
were changed. This is an evaluation, not release qualification or a request
to retire the separately published fastlmmrslib API.

## Candidate, baseline, and scope

The [maintainer's explanation](https://github.com/abhranildas/gx2-py/issues/1#issuecomment-6888043219)
describes an oscillatory integration fix and explicit failure flags. A
[later comment](https://github.com/abhranildas/gx2-py/issues/1#issuecomment-6895340090)
identifies the latest release as containing the fix.

- Candidate: published [gx2 1.6](https://pypi.org/project/gx2/1.6/), uploaded
  November 15, 2026, Python >=3.10, MIT license, universal Python wheel.
  SHA-256: `92efd3c2732d1b3686e2878445807885bd0e1d05d93242426cc17cceccf7485f`.
- Baseline: published `fastlmmrslib==0.1.1`, rather than the earlier C++ backend
  or an unpublished sibling build.
- Linux x86-64 under WSL2; CPython 3.15.2, NumPy 2.5.4, SciPy 1.18.1,
  mpmath 1.4.1. Integration dependencies came from published `fastlmm==1.0.0`
  and `pysnptools==1.0.1`; FaST-LMM integration code was loaded from this worktree.
- Additional numerical screen: CPython 3.11.17, NumPy 2.4.6, SciPy 1.17.1,
  mpmath 1.4.1, the same candidate and Rust versions.
- Only FaST-LMM's score-test adapter imports fastlmmrslib in production:
  `Sc.pv_davies_eig` evaluates an upper tail with unit degrees of freedom,
  zero noncentralities, and no normal component. `eigh` can return tiny negative
  eigenvalues; the experiments preserve them.

## Independent numerical checks

Reran the historical 71-distribution harness against the published Rust wheel,
with both gx2 defaults and `AbsTol=1e-14, RelTol=1e-12`. It includes 45
reference cases, 24 synthetic positive spectra, and two historical spectra.
The latter are read as literal fixture data from fastlmmclib; no sibling
implementation is imported. Rust faults retain a missing probability instead
of being treated as valid reference answers.

Both gx2 settings pass 44 of 45 reference cases at the original `1e-7`
absolute screening threshold. The remaining case is an all-zero spectrum at
zero, which returns NaN. The historical harness expects strict survival zero
there; the existing Rust implementation uses its documented boundary convention
and returns one. A replacement must deliberately preserve FaST-LMM's chosen
boundary behavior. The Python 3.11 tight screen gives the same 44/45 result.

For `P(sum(w_i * Z_i**2) > 20)` with independent standard normals:

| Weights | Independent reference | gx2 defaults | gx2 tight |
| --- | ---: | ---: | ---: |
| `[0.001, 1]` | 7.748269578668e-6 | 7.749421642722e-6 | 7.748269579555e-6 |
| `[0.001, 0.001, 1, 1]` | 4.544537513762e-5 | 4.543167770676e-5 | 4.544537514140e-5 |
| `[1e-8, 1]` | 7.744216471544e-6 | 7.745371946533e-6 | 7.744216472316e-6 |
| `[1e-8, 1e-8, 1, 1]` | 4.539993021648e-5 | 4.540083973859e-5 | 4.539993021768e-5 |

All four return false error flags and no warnings. References use the
independent angular integral or exponential convolution described in the
historical assessment. Passing these checks confirms the reported fix;
it does not establish relative accuracy for arbitrary small probabilities.

A further 60-call analytic probe covered duplicated weights, thresholds
0 through 100, scales `1e-6`, `1`, `1e6`, and selected `tail` and `ray` calls:

- For `[0.001, 0.001, 1, 1]` at 60, the exact tail is
  `9.366989958799e-14`; tight auto returns `9.187095528773e-14` with a false
  error flag, about 1.9 percent relative error. This remains well within
  FaST-LMM's existing absolute tolerance.
- At 100 the exact tail is `1.930680528492e-22`; auto returns a flagged zero.
  Rust at `acc=1e-7` also returns zero, with fault zero. Neither result proves
  tail-relative accuracy; Rust is not used as an oracle here.
- Explicit `tail` gives the correct analytic answers in these duplicated-weight
  far-tail examples. On the three flagged real calls it gives approximately
  `1.1611e-20`, without an error estimate. Those real-call estimates have not
  been independently qualified, so they are not an accepted fallback.
- All six attempted four-dimensional `ray` calls raise `TypeError` inside gx2
  `_ray.py` when converting an array to a scalar with NumPy 2.5.4. No upstream
  patch was applied; this route is not currently a working fallback in the
  tested environment. Monte Carlo alternatives were not qualified.
- Near the positive support boundary some analytically correct values are
  nevertheless flagged as clipped. An adapter cannot equate a true flag with
  a known magnitude of error.

The ordinary 12-call serial/four-thread repeatability probe also passes.
It does not qualify every gx2 method's concurrency behavior.

## Integration, diagnostics, and speed

The process-local experiment substitutes `Sc.pv_davies_eig`, runs both
SNP-set score tests and all four canonical Davies input tests, and captures
138 probability calls. Production files and expected results are unchanged.

| Backend policy | Result |
| --- | --- |
| Published Rust baseline | 6 passed |
| gx2 tight, record diagnostics but accept returned probabilities | 6 passed |
| gx2 tight, reject invalid probabilities, true boolean flags, or numeric error estimates above `1e-7` | 3 passed, 3 errors |

Errors occur in `test_four`, the two-kernel linear Davies input test, and
its reordered-ID variant. The flagged calls have 47 eigenvalues including
negative roundoff terms, a statistic near 137.63, and return zero. Baseline
Rust also returns zero, but reports fault zero under its absolute-error
contract. Ignoring gx2's flag would discard the diagnostic improvement that
motivated the upstream fix. Whether a zero is adequate here requires an
explicit accuracy contract or an independently justified bound.

The maximum absolute probability difference across the permissive capture is
`1.4315419905042148e-8`. All three warnings in that capture concern clipping.

Three sequential replay passes over the same 138 preconstructed inputs,
with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`, give:

| Backend | Median summed kernel time | Relative to Rust | Flagged calls per pass |
| --- | ---: | ---: | ---: |
| Rust, `acc=1e-7` | 0.002363 s | 1 | 0 |
| gx2 auto defaults | 0.330206 s | 140 | 3 |
| gx2 auto tight | 0.654779 s | 277 | 3 |

These measure the caller-visible scalar kernels, including gx2 parameter
array creation. They exclude the rest of the association workflow and are
not a platform-wide benchmark. The original instrumented integration capture
summed to 0.00555 s Rust and 0.70686 s gx2 tight. Both baseline and candidate
integration runs intentionally evaluate both kernels, so their total suite
wall times do not measure a production end-to-end speed difference.

The unmodified focused Rust-adapter regression tests pass: five tests and six
subtests. Legacy integration tests changed a bundled executable's mode; its
original mode was restored afterward.

## Reproduction and remaining requirements

Reusable drivers, frozen environments, metadata, and detailed results are in
`/mnt/c/deldir/gx2-evaluation` (`C:\deldir\gx2-evaluation`). Virtual environments
are in `/home/carlk/.cache/gx2-evaluation/py315` and `py311`. The external
`run_evaluation.py` reads the worktree's historical scripts and adapts them in
memory to the Rust import, missing-probability contract, and chosen settings.
It does not alter those tracked scripts. Its strict mode enforces diagnostic
checks; the original historical integration script does not.

```bash
PY=/home/carlk/.cache/gx2-evaluation/py315/bin/python
DRIVERS=/mnt/c/deldir/gx2-evaluation
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PY" "$DRIVERS/run_evaluation.py" numerical
"$PY" "$DRIVERS/run_evaluation.py" numerical --tight
GX2_BASELINE=1 GX2_EVALUATION_OUTPUT="$DRIVERS/baseline-rerun" \
    "$PY" "$DRIVERS/run_evaluation.py" integration
GX2_EVALUATION_OUTPUT="$DRIVERS/candidate-rerun" \
    "$PY" "$DRIVERS/run_evaluation.py" integration --tight
GX2_EVALUATION_OUTPUT="$DRIVERS/strict-rerun" \
    "$PY" "$DRIVERS/run_evaluation.py" integration --tight --strict
"$PY" "$DRIVERS/analytic_sweep.py"
"$PY" "$DRIVERS/replay.py"
```

Run integration variants sequentially because legacy tests overwrite their
output files. The strict run is expected to exit nonzero with three errors.
`requirements-py315.txt`, `requirements-py311.txt`, `environment-py315.json`,
and saved PyPI JSON record the exact tested artifacts and dependencies.

Before adopting gx2, implement and independently qualify a domain-specific
adapter: validate finite inputs and outputs, preserve degenerate boundary
behavior, interpret method-dependent diagnostics, and establish a reliable
policy for flagged and small tails without silently dropping eigenvalues,
clipping outputs, or presenting unbounded approximations as validated p-values.
Then decide whether removal of native-package maintenance justifies the
workload-specific performance cost.

A replacement would still need the complete FaST-LMM suite, installed wheel
and sdist tests, Python 3.12–3.14, other supported OS/architecture combinations,
minimum dependencies, and optional-feature configurations. None of those was
run for this candidate. With no production change and unresolved adapter
requirements, this assessment stops before that release-qualification matrix.
No upstream comments were posted and nothing was published.
