# Quadratic-form replacement evaluation

Assessment date: November 13, 2026. Related proposal:
[FaST-LMM #63](https://github.com/fastlmm/FaST-LMM/issues/63).

## Decision

Do not adopt the published `gx2==1.5` as FaST-LMM's quadratic-form backend
for 1.0.0. Changed numerical results are acceptable for this release, but
the candidate must still compute accurate probabilities. This assessment
found substantial errors against elementary formulas and an independent
angular integral, including inputs in FaST-LMM's actual central, positive
weighted-chi-square domain. The finding does not establish that every gx2
method is unsuitable or that the existing C++ always gives the correct answer.

The recommended next implementation experiment is the issue's Rust fallback:
port the existing Davies implementation into a reentrant Rust core in
fastlmmclib, preserving its independently useful generic distribution API.
Qualify it against independent answers, not merely the C++ output. This
recommendation is not an implemented or qualified replacement. A later
upstream gx2 fix could also be reevaluated against the same cases.

No production backend, dependency requirement, expected result, or tolerance
was changed during this assessment. Retirement of the bundled `fastlmmc`
executables and their callers remains a separate compatibility change.

## Candidate and environment

- [gx2 1.5](https://pypi.org/project/gx2/1.5/), released August 31, 2026;
  Python >=3.9, MIT, universal Python wheel.
- Wheel SHA-256: `a8a619976653c9c8de8c7be4663f454f9ce2054a973d4329f8c66be68674ee1b`.
- [Upstream source](https://github.com/abhranildas/gx2-py): last reported push
  August 31, 2026; no open issues when checked. Absence of reported issues
  is not evidence of numerical correctness.
- CPython 3.15.2, Linux x86-64 under WSL2, NumPy 2.5.4, SciPy 1.18.1,
  mpmath 1.4.1.
- Baseline: prepared fastlmmclib 1.0.0 installed wheel, with the existing
  QFC algorithm unchanged. These are experimental comparisons, not final
  release acceptance against published prerequisite artifacts.
- [chi2comb](https://pypi.org/project/chi2comb/) was rechecked: latest
  release remains 0.1.0 from September 2020, with wheels ending at CPython
  3.7. It is not a ready binary dependency for this Python 3.15 upgrade.

## Independent correctness checks

The expanded harness evaluated 71 distributions: 45 with elementary-formula or
SciPy-distribution reference answers, 24 synthetic positive spectra with
ranks 2/10/50/200 and condition ranges up to 1e8, and the two existing
fastlmmclib regression inputs. Some simple gx2 paths themselves use SciPy;
those checks validate dispatch and parameter mapping, rather than provide
algorithmically independent validation. Fourteen of the 45 reference cases
produced a non-finite result or absolute error greater than 1e-7 with
`method='auto'`. This screening threshold matches FaST-LMM's existing
requested absolute accuracy; it does not qualify relative accuracy in tiny
tails. Failures included mixed signs, zero padding, strongly unequal positive
weights, and the all-zero distribution at its boundary.

Additional checks used only unit degrees of freedom and zero noncentrality,
exactly as in `Sc.pv_davies_eig`. Let independent `Zi` be standard normals
and evaluate `P(sum(wi * Zi**2) > 20)`:

| Weights | Independent reference | Existing QFC, acc=1e-7 | gx2 auto |
| --- | ---: | ---: | ---: |
| `[0.001, 1]` | 7.748269578667993e-6 | 7.74698446570632e-6 | 0.003614705553654729 |
| `[0.001, 0.001, 1, 1]` | 4.544537513762248e-5 | 4.544490645874699e-5 | 0.0008146527586513352 |
| `[1e-8, 1]` | 7.744216471543636e-6 | 7.741647782055772e-6 | 0.0011529390032550824 |
| `[1e-8, 1e-8, 1, 1]` | 4.539993021648416e-5 | 4.539953405613417e-5 | 0 |

All four QFC calls reported fault=0. All four gx2 calls emitted integration
warnings; the last also emitted a clipping warning. The C++ errors in this
table are within its requested absolute tolerance, unlike gx2's errors.

For duplicated weights `[a, a, 1, 1]`, the distribution is the sum of two
independent exponentials, giving the elementary survival function
`(exp(-x/2) - a*exp(-x/(2*a))) / (1-a)`. For the two-weight case, polar
coordinates give the independent reference
`(2/pi) * integral(exp(-x/(2*(a*cos(theta)**2 + sin(theta)**2))), theta=0..pi/2)`.
It was evaluated with SciPy quadrature at abs=1e-14, rel=1e-12. These reference
calculations do not use QFC or gx2's characteristic-function inversion.

## Method settings and diagnostics

The published automatic dispatcher tries Ruben's series for same-sign
weights and falls back to Imhof integration when necessary. Tiny negative
roundoff eigenvalues or exact zeros can also route directly to Imhof.
Scalar basic integration uses SciPy `quad`'s default 50-subdivision budget;
its estimated integration error is discarded. The returned Imhof error
output describes clipping, not a rigorous probability error bound.

Tighter absolute/relative tolerances did not repair the difficult cases.
mpmath integration at 30 and 60 digits also failed to give reliable answers;
at 60 digits it could return an inaccurate result without a warning.
An experimental, process-local change to the quadrature subdivision budget
of 1,000 and 10,000 still produced convergence/roundoff warnings. At 10,000
subdivisions, `[0.001, 1]` returned 3.8136693476797934e-5 instead of
7.748269578667993e-6 and took about 3.23 seconds. This patch was solely an
evaluation experiment; it is not a proposed dependency monkeypatch.

The current C++ is not an unconditional oracle either. Ten main-harness
calls reported nonzero faults when asked for acc=1e-10; those outputs must
not be interpreted as successful reference probabilities. Very small tails
also require further independent validation. The existing 1e-12 regression
comparison is a historical output fingerprint, not a proven 1e-12 truth
guarantee when the algorithm was asked for acc=1e-7.

gx2's public return contract differs from QFC: there is no equivalent
Davies fault/trace tuple, and some explicitly selected tail methods return
negative base-10 log probabilities below floating-point range. A production
adapter would need documented semantics and must not fabricate successful
fault codes or silently reinterpret such outputs as ordinary probabilities.

## Actual FaST-LMM integration and performance

The integration experiment temporarily replaced `Sc.pv_davies_eig` in the
test process, with no production source change. It exercised both SNP-set
score tests and all four canonical Davies input-file tests, including
logistic and reordered-ID inputs.

- Existing QFC baseline: all six tests passed.
- gx2 substitution: three passed, three failed existing result-order checks
  (`SetId` column mismatch). Expected outputs were not changed.
- Captured calls: 138, including repeated cases; maximum absolute p-value
  difference 6.074297925762195e-8; four calls emitted gx2 warnings.
- Summed kernel timings: gx2 0.4586 seconds versus QFC 0.00738 seconds,
  approximately 62 times slower in this small sample. This is not an
  end-to-end performance claim or a statistically robust benchmark.
- Three ordering failures are consistent with small numerical changes near
  tied/zero tails. Since 1.0 permits numerical changes, the ordering failures
  alone are not grounds to reject a candidate. The independent analytic
  failures are the decisive evidence here.
- Twelve ordinary serial calls and twelve calls across four threads were
  identical. This checks basic-path repeatability only, not the safety of
  every method or concurrent manipulation of mpmath precision.

## Reproduction and remaining work

The assessment scripts are [evaluate_gx2.py](experiments/evaluate_gx2.py)
and [integration_gx2.py](experiments/integration_gx2.py). The first reads the
two reference spectra from the sibling fastlmmclib test fixtures. Install
the prepared fastlmmclib wheel and exact candidate/dependency versions into
an isolated Python 3.15 environment before running it. The integration
script additionally needs the project's installed dependencies and source
test data. Run it once normally for gx2 and once with `GX2_BASELINE=1` for
QFC; each run overwrites the normal test output files. Run these sequentially.
`GX2_EVALUATION_OUTPUT` chooses the JSONL capture directory, defaulting to
`/tmp/fastlmm-maintenance-review`. These scripts are experiments, not ordinary
CI or final artifact qualification against sibling source checkouts.

Detailed JSONL data, tail checks, increased-subdivision results, and both
integration logs are retained in `/tmp/fastlmm-maintenance-review` for this
session. No upstream issue/comment was posted and no release was published.
The verified, self-contained [upstream issue draft](GX2_UPSTREAM_ISSUE.md)
contains two elementary-reference reproducers and a working diagnosis.

Because independent correctness checks failed, this assessment stopped
before a complete OS/architecture/Python 3.11-3.15 qualification, minimum
dependency tests, all alternative gx2 methods, full FaST-LMM suite under the
candidate, generic external-user characterization, or installed FaST-LMM
wheel/sdist acceptance with the replacement. None is claimed to have passed.
Those remain required if an improved candidate is selected. A Rust port
also needs failure/diagnostic tests, concurrency checks, numerical and
performance qualification, provenance review, and platform wheel validation.
