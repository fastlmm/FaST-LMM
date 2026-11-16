# GPU documentation and API review

<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

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
