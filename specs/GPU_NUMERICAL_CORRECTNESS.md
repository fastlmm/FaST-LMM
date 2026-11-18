<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

# GPU alpha numerical correctness investigation

Investigated November 18, 2026 in `investigate/gpu-alpha-numerics`, based on
FaST-LMM `d37dccf`, PySnpTools `75652ce`, and bed-reader `71f62e4`.
The explicitly identified development imports, binary provenance, exact commands,
full comparisons, tests, and unresolved failures are recorded in the
[qualification report](/mnt/c/deldir/fastlmm-correctness-20261118/REPORT.md).
This is development-code evidence, not published-alpha or release acceptance.

- **Corrected:** epistasis multiplied float32 SNPs before promotion into its
  float64 design. For a redundant interaction this injected a spurious likelihood
  gain. Promote before multiplication; float64 results remain identical.
  All 1,225 notebook pairs now pass the unchanged `1e-5` checks: max absolute P
  error `2.4410e-6`, down from `2.7787e-5`. The new regression fails against the
  original implementation and passes after the correction for both dtypes.
- **Explained float64 fitting roundoff; no correction recommended:** independent
  float64 h2 fits differ near the likelihood rounding floor. Synthetic maximum
  delta log10(P) is `5.3600e-8` against the original strict `2e-8` diagnostic;
  mouse is `1.7295e-7`. These measured exceedances remain recorded. Shared CPU64
  h2 reduces them to `2.4869e-14` and `2.1316e-13`.
  Follow-up long-double derivative reference fits show ordinary h2 errors below
  `4.982e-8` and likelihood loss below `3.948e-13`; the largest relative P error
  against those reference fits is `3.055e-7` (about `0.000031%`). Independently
  fitted reference CPU/GPU models agree to about `3e-13` in log10(P).
  This establishes optimizer roundoff as the cause, not a wrong backend
  association calculation. Retain the documented limitation and current
  optimizer; no tolerance or expected output was changed. This is a technical
  recommendation, not a claim of additional human approval.
- **Accepted alpha precision limitation:** on November 18, 2026, Carl accepted
  the measured mouse GPU32 differences with a user warning and concrete example.
  This resolves the mouse-specific accuracy decision, not the other numerical
  gates or Carl's broader review. Mouse CuPy32 max delta log10(P)
  is `3.7262e-4`, or `3.9958e-4` with CPU64 h2 fixed. Native float32 GPU
  eigensolver error dominates: double decomposition on the same float32 kernels
  reduces the two worst chromosome errors to about `1e-5`. The existing
  `1.5e-4` backend bound and earlier `1.6e-4` triangle bound remain failed.
  These exceedances remain recorded as an approved alpha precision exception;
  they are not passing strict comparisons. No silent float64 solve or tolerance
  widening was introduced. The GPU guide and single_snp dtype documentation warn
  users with CPU64 P=`2.0918834e-87` versus GPU32 P=`2.0899597e-87` at fixed h2
  (about 0.092% relative error); this is not a universal error bound.
- **Epistasis now requires CPU float64:** full float32 preprocessing still
  exceeded legacy fixture tolerances after the product correction. The public
  function now warns on non-NumPy/float64 defaults and computes in CPU float64,
  including explicit worker reads; caller settings are restored on success and
  failure. Float32 eigen caches are rejected with recomputation instructions.
  This is an explicit execution contract, not a correction of native float32
  epistasis. All old assertions and original failure evidence are retained.
  Precision-contract regressions cover actual computation, warning counts,
  raw in-memory float32 inputs, error restoration, cached models, and a real
  subprocess with conflicting worker defaults.

The three requested findings are now diagnosed and addressed for the qualified
configurations. The independent-fit strict diagnostic remains a documented
precision limitation; the mouse GPU32 exception has Carl's explicit acceptance;
epistasis uses CPU float64. Full notebook qualification, CPU thread benchmarks,
Carl's broader review, installed-artifact checks, and release preparation remain
open in [todo.md](todo.md). No publication or merge is authorized here.

The numerical worktree is now the canonical combined API-policy change. Final
canonical validation runs 227 tests with no failures; the five optional HDF5
skips all pass separately. Combined regressions pass 60 tests and 94 subtests.
See [the combined status](GPU_API_POLICY_STATUS.md) for final boundary checks,
source hashes, and the preserved four-notebook handoff. Numerical comparison of
those saved notebook runs remains subsequent work; Carl's review is not complete.
