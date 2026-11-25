<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

# GPU alpha numerical correctness investigation

Investigated November 18, 2026 in `investigate/gpu-alpha-numerics`, based on
FaST-LMM `d37dccf`, PySnpTools `75652ce`, and bed-reader `71f62e4`.
The bed-reader reference was the earlier GPU implementation; bed-reader is now
CPU-only, so this record is historical numerical evidence rather than
qualification of the current CPU-read/CuPy-transfer path.
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

The numerical worktree's combined API-policy change was merged into
`gpu-bed-reader`. Its canonical validation ran 227 tests with no failures; the
five optional HDF5 skips all passed separately. Combined regressions passed 60
tests and 94 subtests. These results validate that historical development
snapshot, not the current CPU-read/CuPy-transfer path. The four saved notebook
runs used bed-reader's previous GPU path and are historical evidence; rerun the
configurations against current heads as tracked in [todo.md](todo.md). Carl's
review remains open.

## Detail moved out of the user GPU guide (November 25, 2026)

The user guide `doc/source/gpu.rst` was shortened from about 3,200 to about 1,100 words. Its numerical-qualification section is preserved verbatim below. The full previous guide, including the GPU BED read example that observes `cupy.linalg.eigh`, the `LocalMultiProc` worker example, the small CPU/GPU comparison example, and the `single_snp` options table, is at `git show 1fccabc:doc/source/gpu.rst`.

```rst
Numerical qualification limits
------------------------------

Independent variance-component fits can differ even in float64. Tiny backend
roundoff changes which h2 is selected when likelihood values near the optimum
are indistinguishable at floating-point precision. A strict comparison of
independently fitted p-values therefore also measures optimizer sensitivity.
For a backend arithmetic diagnostic, fit h2 on the CPU and pass that same h2 to
both calls, matching the background kernel and chromosome exclusions. Retain
both the independent-fit results and this diagnostic; passing the latter does
not make the former pass.
An independent long-double likelihood-derivative diagnostic on the synthetic
and mouse datasets found the ordinary float64 fits within approximately
``5e-8`` in h2 of the reference optima, with likelihood loss below ``4e-13``.
The largest relative P-value difference from those reference fits was about
``3.1e-7`` (``0.000031%``). This explains the independent-fit roundoff without
changing the optimizer or a test tolerance; it is an observed example, not a
universal error bound.

.. warning::

   GPU float32 results can differ more from CPU float64 than CPU float32 results
   do. For the mouse dataset (1,940 samples and 10,150 SNPs), the largest observed
   difference was approximately 0.0004 in log10(p), or 0.09% relative P-value
   error. With the same fitted h2, SNP ``rs13479446`` had P-value
   ``2.0918834e-87`` on CPU float64 and ``2.0899597e-87`` on GPU float32.
   Both are extremely small, but their relative difference is about 0.092%.

   This example is an observation, not an upper bound for other datasets or
   hardware. Check numerical differences on representative data, including
   associations near a decision threshold. Use float64 when closer agreement
   is required. Float64 statistics computed after a float32 decomposition
   cannot recover precision already lost in that decomposition.

The same array dtype does not guarantee identical solver results. NumPy and
CuPy both receive float32 matrices and return float32 eigenvalues and
eigenvectors, but their LAPACK and cuSOLVER implementations and reduction orders
can round differently. For the 1,940-sample mouse example, CuPy float32 differed
from NumPy float64 by up to approximately 0.0004 in log10(p), even with h2 held
fixed. A diagnostic that explicitly promoted the same GPU float32 kernel to a
double-precision solve reduced the errors on the two worst chromosomes to about
0.00001. These are measured examples, not universal error bounds. Float64 is
needed when the observed float32 errors exceed a study's accuracy requirement;
check all association results, including very small p-values and near-null ties.
Agreement of the strongest associations alone is insufficient.

Epistasis requires CPU float64 computation. If ``ARRAY_MODULE`` or
``ARRAY_DTYPE`` requests another configuration, ``epistasis`` emits a
``UserWarning`` naming the requested settings and the actual CPU float64
configuration, then restores the caller's settings on return or failure.
Reader operations in workers explicitly use NumPy float64 as well. Float32 input
arrays are widened, but this cannot undo their original quantization. An
existing float32 eigen cache is rejected: remove that cache or use a new cache
path to recompute it in float64. Native float32 epistasis is unqualified; its
legacy fixture failures were preserved as evidence rather than weakening tests.

```
