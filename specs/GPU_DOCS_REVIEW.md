# GPU documentation and API review

<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

## Handoff: November 16 development snapshot

This section records the November 16 development snapshot; its checkout and
working-tree status are historical. bed-reader switched to CPU-only on
November 21, so this audit covers the PySnpTools and FaST-LMM GPU APIs. The
current review heads and remaining notebook qualification are tracked in
`todo.md` and `REVIEW_INVENTORY.md`.

All three audit findings now have implementation fixes:

- Python CPU Beta constant columns: zero their weights before multiplication.
- FaST-LMM concurrent backend selection: resolve under the workflow lock.
- Blocked BetaTrained kernels: validate the full SNP order and reuse each
  block's corresponding moments without refitting or relaxing public checks.

All three fixes passed relevant validation. The latest full canonical
PySnpTools run passed all 750 tests on CPython 3.13 with the matching development
CUDA environment. The third fix also passes CPU/GPU regressions, Python 3.14
CPU checks, native BED integration checks, all guide examples, strict Sphinx
builds, and Ruff. The durable full-suite output is
`~/.cache/python-docs-audit/trained-beta-canonical.log`. No checks remain running.

## Current-source Python documentation audit (November 16, 2026)

This audit reads the primary PySnpTools and FaST-LMM source checkouts, including
uncommitted GPU implementation work. The historical review below is not the
source of truth for the current API. Runtime Python ASTs are checked against a
snapshot taken at the start of this audit; changes made by this audit are
restricted to documentation and docstrings, preserving existing line endings.
No bed-reader implementation, dependency bounds, lockfiles, or release artifacts
are changed.

### Documentation changes

- Keep README GPU sections brief and link to one complete guide per project.
  The earlier preview URL remains a short pointer to the current PySnpTools
  guide; it no longer describes an unimplemented or separate API.
- Correct landing pages, default/option tables, and generated signatures, and
  link reader, standardizer, trained-standardizer, and association references
  to the full examples. Register FaST-LMM's documented package with Sphinx so
  its module-index link points to an emitted page, and fix its issue-tracker URL.
- Explain explicit `xp` versus `ARRAY_MODULE`, strict missing-CuPy errors,
  actual-device standardization, per-read dtype/order choices, CPU-only versus
  GPU-ignored arguments, and native GPU BED build/runtime requirements.
- Correct BED's float64 default and unsupported float16 claim. Document the
  int8 opt-in and missing sentinel, which do not make int8 standardizable.
- Explain observed population variance, post-imputation variance, constant and
  wholly missing columns, applying versus fitting moments, host trained stats,
  UnitTrained ID remapping, and BetaTrained's exact alignment requirement.
- Correct the claim that CuPy Beta evaluates the density on the GPU: the current
  code transfers per-SNP means to SciPy on CPU and then returns weights to CuPy.
- Explain float64 CPU training-kernel formation even for float32 LMMs,
  float32 rounding in EffectSize moments, GPU indexing/memory limits, cache
  input responsibilities, result columns/row counts, and worker placement.

### Implementation/API findings

1. **Pure Python CPU Beta can turn constant columns into NaN (fixed November 16, 2026).**
   `SnpData` with constant zero/two columns standardized by `Beta(0.5, 0.5)`
   yields zeros with the native implementation and NaNs with
   `force_python_only=True`. The singular endpoint PDF is multiplied by the
   centered zero, and fresh Python training does not zero the constant columns
   afterward. This also affects automatic Python fallback cases. The fix sets
   constant-column weights to zero before multiplication, avoiding invalid
   operations during fitting and trained application. A regression in the
   canonical PySnpTools suite (TestBetaPython) covers both float dtypes, C/F orders, singular
   and nonsingular Beta parameters, missing values, trained constants with
   varying new values, and automatic fallback from a larger contiguous buffer.
   The regression failed on the original code and passes with the fix. This is
   a long-standing defect: the same logic exists in June 2015 commit 8dbe18d3.
   The earlier docstring-only audit statement above predates this authorized fix.
   Follow-up validation passed: canonical PySnpTools suite, 748 tests; the
   regression with 16 parameter subtests on Python 3.13 and 3.14; 22 focused
   CPU/native/GPU standardization tests with 16 subtests; and the strict Sphinx
   HTML build. The standalone in-memory regression is registered in both
   canonical unittest and pytest paths without depending on BED fixture setup.
2. **Implicit workflow backend resolution races with a concurrent workflow (fixed November 16, 2026).**
   `single_snp` resolves `xp` before acquiring `_array_module_lock`. With an
   initial `ARRAY_MODULE=numpy`, a first explicit CuPy call can establish its
   temporary environment while a second `xp=None` call resolves to CuPy. The
   second call runs after the lock is released but retains that selection.
   Reproduced deterministically using event signals, with the existing
   implementation and a stub for expensive numerical work. Explicit `xp`
   avoids this ambiguity; restoration itself succeeds in the reproduction.
   This is long-standing process-wide environment behavior, present on main and
   in October 2020 commit e882912. The earlier lock improved coordination but
   left selection outside its protected region. The fix moves backend selection,
   two-kernel NumPy coercion, and the temporary environment into one private
   context manager holding the same lock throughout. Omitted xp now resolves
   after preceding workflows restore the environment. Four event-coordinated
   regressions cover NumPy/CuPy first calls and second calls with/without a
   second kernel; three failed on the original code and all pass with the fix.
   CPU-only cases also pass on Python 3.14 without requiring native GPU setup.
   Final validation passed: all 86 tests in the complete single_snp suite plus
   worker/environment and new backend-selection tests on CPython 3.13 with
   CUDA; both CPU-only regressions on CPython 3.14; and the strict Sphinx HTML
   build. Existing deprecation, memory-target, and complex-cast warnings were
   reported without adding suppression rules.
3. **Blocked kernels fail with BetaTrained despite matching full SNP IDs (fixed November 16, 2026).**
   Training `Beta(1, 25)` on six SNPs and then using
   `fresh.read_kernel(trained, block_size=2)` raises the trained-ID assertion;
   the unblocked call succeeds. Each block contains only a subset of IDs, but
   BetaTrained requires the entire training list. The fix validates that full
   list and the stats shape before blocking, then constructs a block-local
   BetaTrained from positional slices of the matching IDs and moments. This
   preserves distinct moments even for repeated IDs. The public standardizer's
   strict alignment checks are unchanged, no moments are refitted, and internal
   return_trained calls return the original trained object rather than attempting
   an unsupported merge. The guide now includes a tested blocked/unblocked
   trained-kernel example. The defect was reproduced before applying the fix.
   Final validation passed: 750 canonical PySnpTools tests on CPython 3.13;
   three focused CPU/GPU tests with 64 parameter subtests; the CPU tests on
   CPython 3.14 with 40 subtests; all 21 GPU API tests including eight new native
   BED cases; seven complete guide examples; strict Sphinx builds; and Ruff.
   Tight blocked/unblocked comparisons use floating-point roundoff bounds.
   A separate independent reference check allows the measured pre-existing
   native statrs versus SciPy Beta(1,25) PDF difference (about 1.5e-13 relative
   for the selected mean), verified against the analytic 25*(1-p)**24 density.
   No existing numerical expected results were changed.

The former missing-CuPy fallback, reader-`xp` omission, Unit device mismatch,
and fractional-index truncation findings in the historical review have already
been addressed by the existing uncommitted implementation. They are not
reported as remaining bugs by this audit.

### Current audit validation

- Seven independent, self-contained guide examples passed on the local CUDA
  GPU with CPython 3.13. They create their own synthetic data and verify actual
  NumPy/CuPy array types, dtype/order, trained moments and alignment, native
  BED results, blocked kernels, cache reuse, and real worker-process reads.
  FaST-LMM's BED example observes the actual CuPy eigensolver input.
- Relevant existing tests passed: PySnpTools 41 tests plus 992 subtests;
  FaST-LMM 31 tests, including all 11 real backend/environment tests. This is
  a focused selection, not a claim that the full suites were rerun.
- API doctests passed: 38 in edited PySnpTools modules and one FaST-LMM module
  doctest. Precision, raw-array-deprecation, and existing numerical-cast warnings
  were reported without introducing warning filters.
- Both complete Sphinx HTML builds pass with `-W --keep-going -E -a` on
  CPython 3.14. Generated guide tables, code blocks, and navigation were visually
  inspected in Chromium; rendered reader/association signatures and all local
  links/fragments from the landing pages and guides were checked against current
  source. No documentation is published.
- Reusable runners and deterministic bug reproductions are in
  `/mnt/c/deldir/python-docs-audit/`. The example runner executes each block in
  an independent namespace so missing imports or dependencies on earlier
  examples cannot be hidden.
- Generated HTML and test outputs are under
  `~/.cache/python-docs-audit/{PySnpTools,FaST-LMM}/html` and adjacent cache
  directories. Repository-generated HTML is not edited or checked in.
- Not qualified here: the complete Python/OS matrix, HIP, other GPU hardware,
  near-capacity/large-matrix performance, release wheels/sdists, published
  dependency acceptance, or external-link availability. Native GPU examples
  intentionally use the existing matching development artifact environment.

## Earlier integration status

The documentation commits `62f2476` and `0d74f4a` have been cherry-picked into
the primary development checkouts alongside their existing GPU API changes.
The guides now describe explicit reader backends, standardization on the actual
array device, native GPU BED reads, strict missing-CuPy/build errors, and worker
backend handling. The duplicate `single_snp` backend parameter documentation is
consolidated. The former preview is now labeled as implemented development API
examples, with published-release availability still explicitly qualified.

Integration validation on CPython 3.13 passed both complete Sphinx HTML builds
with `-W --keep-going -E -a` and all ten GPU guide examples against the primary
source checkouts. The BED/phenotype examples used synthetic local data on the
NVIDIA CUDA GPU. Python runtime ASTs match the saved pre-merge implementation;
source line endings and all previously untracked files are preserved. The
expected float64-to-float32 warnings were reported. This is documentation
integration validation, not published-artifact or full-platform acceptance.

The review below records findings against the earlier committed implementations.
Findings 1 and 2 are addressed by the concurrent implementation; finding 4's
fallback behavior is replaced by an ImportError, and workflow serialization is
implemented. Finding 3 (fractional low-level indices being cast before validation)
was subsequently fixed in PySnpTools: both NumPy and CuPy now validate integer
index types, shape, and bounds before converting to the native index dtype.
Regression coverage includes floating-point, Boolean, nonfinite, oversized, and
negative indices, plus valid duplicates and empty selections. The full canonical
PySnpTools suite passed all 747 tests. Five focused CPU/CUDA subsetting tests also
passed against the installed wheel without source imports, and the updated
index-rejection doctest and strict Sphinx build passed.

## Original review

Reviewed November 16, 2026, against FaST-LMM `e0f81a9` (`gpu-bed-reader`)
and PySnpTools `d647dbc` (`cupy-sub-matrix`), including their changes since local
`main`. Documentation edits are isolated on `docs/gpu-features` in each repository.
The original checkouts contain concurrent uncommitted API work; it was inspected
but was not copied into these branches. No bed-reader files were changed.

## Documentation coverage

The GPU features were not discoverable from either committed README or Sphinx
landing page. FaST-LMM's `dtype` parameter already explained float32, but GPU
setup, native BED prerequisites, memory limits, cache behavior, and examples
needed a user guide. PySnpTools needed examples for device transfers, dtype
preservation, CuPy subsetting, and standardization. Its release notes already
covered the non-GPU Hashdown retry and sample-manifest changes since `main`.

The main guides describe the committed implementations. A separate
PySnpTools API preview, plus a FaST-LMM integration section, covers the
concurrent implementation without claiming it is already committed. FaST-LMM's notebooks in
`specs/experiments/` remain experimental evidence; their timing tables are not
used as general performance promises. Docstring edits correct `sub_matrix`'s
reversed C/F layout descriptions and remove an incorrect return-type field
from `single_snp`'s backend parameter documentation.

## Confirmed API findings

1. **Reader backend selection does not retain the constructor's explicit choice.**
   With `ARRAY_MODULE=numpy`, constructing `SnpData(..., xp=cupy)` and then
   calling `data[[0, 2], :].read(dtype="float32")` returns NumPy values. An
   explicit constructor argument looks like a persistent device choice, but the
   read constructs another object using the environment. Reproduced on the GPU.
   The committed guide demonstrates the necessary environment context.
   Concurrent code adds explicit `xp` arguments to reader methods; after that
   code lands, prefer explicit reader arguments in the guide and verify both
   environment overrides and existing-array behavior.

2. **Unit standardization can contradict an existing array's device.**
   Under `ARRAY_MODULE=numpy`, `SnpData(..., xp=cupy).standardize()` raises a
   `TypeError` about implicit conversion to NumPy. In `standardizer/unit.py`,
   Unit consults the environment and calls `xp.asarray(snps.val)` before the
   standardization helper can dispatch on the actual array. Reproduced on the
   GPU. The guide keeps the environment consistent with the array. Concurrent
   code changes Unit to use the actual array module; once merged, simplify the
   guide and run both directions of the environment/device mismatch tests.

3. **Low-level CuPy index validation silently truncates fractional inputs.**
   `sub_matrix(cupy.arange(6, dtype=cupy.float32).reshape(3, 2), [0.9], [0])`
   returns the element at row zero. `_checked_index` casts to `numpy.intp`
   before validating, so the range check loses the evidence of invalid input.
   The CPU path also casts indices. Recommend validating integer input before
   conversion, with regressions for fractional, Boolean, nonfinite, and
   oversized indices on both devices. This review does not change behavior.

4. **Backend selection still has process-wide and fallback semantics.**
   `array_module("cupy")` can return NumPy after a failed import; `single_snp`
   temporarily modifies `ARRAY_MODULE`. The guide makes fallback visible and
   checks hardware explicitly. Concurrent code adds workflow serialization and
   worker backend handling. Review runner overrides and exception restoration
   when integrating those changes; do not advertise independent concurrent
   CPU/GPU workflows merely because each call takes an `xp` argument.

The concurrent `single_snp` docstring also currently contains two `:param xp:`
fields. Consolidate these when merging the implementation so the new workflow
semantics and existing fallback/two-kernel explanation form one parameter entry.
Do not cherry-pick the docstring hunks without reviewing that overlap.

## Validation and integration

- Both complete Sphinx HTML builds passed with `-W --keep-going -E -a` on
  CPython 3.14, with the worktree sources on `PYTHONPATH`.
- All five PySnpTools and three FaST-LMM guide code blocks ran successfully on
  the local CUDA GPU using CPython 3.13. Synthetic BED and phenotype files stood
  in for the user's `cohort` files. This exercises examples against these branch
  sources, not published-artifact release acceptance.
- Both additional explicit-reader preview examples passed on the GPU against
  the original concurrent development sources, using
  `~/.cache/bed-reader-gpu/qualification/artifact-venv/bin/python`.
- Five existing PySnpTools GPU/subsetting/standardization regressions and seven
  FaST-LMM precision/cache/two-kernel regressions passed (12 total). Existing
  deprecation warnings were reported, not suppressed.
- AST comparison confirms the Python edits change docstrings only, and byte
  checks confirm original line endings are preserved.
- The example runner is `/mnt/c/deldir/gpu-docs-review/check_gpu_docs.py`.
  Set `PYTHONPATH` to both documentation worktrees and run it using the existing
  `~/.cache/fastlmm-gpu/venv/bin/python` environment. For the preview, add
  `--preview`, set `PYTHONPATH` to the original development sources, and use the
  artifact environment above.
- Generated HTML is under `~/.cache/fastlmm-gpu-docs-build/html` and
  `~/.cache/pysnptools-gpu-docs-build/html`; generated output is not committed.
- Existing CRLF files retain CRLF. Use
  `git -c core.whitespace=cr-at-eol diff --check` for FaST-LMM.

When the concurrent API work is committed, promote the validated preview's
explicit `read(..., xp=...)` and `read_kernel(..., xp=...)` examples into the main
PySnpTools guide, remove obsolete environment workarounds, and consolidate the
native BED, strict missing-GPU error, and CuPy Beta descriptions. Update
FaST-LMM's native-reader fallback explanation at the same time. Keep the preview
labeled until those implementation commits are integrated; it describes the
current development sources and has been tested against them.

Full platform/Python matrices, release artifacts, and GPU performance benchmarks
are not qualified by these documentation checks. Prerequisite releases and
published-artifact acceptance remain release work.
