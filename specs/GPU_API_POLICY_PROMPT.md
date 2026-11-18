# Implement the FaST-LMM public API backend and precision policy

<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

Implement the following agreed goal across FaST-LMM's public APIs. This is a library-wide task; scripts, notebooks, and worker processes must receive the same guarantees.

## Agreed goal

- Single-kernel `single_snp` must support CPU/GPU (NumPy/CuPy) × float32/float64, with validated numerical accuracy.
- Two-kernel analyses and other functions need not support GPU or float32. They must either honor the requested configuration, clearly warn about the actual fallback backend and precision, or raise an informative error. They must not silently fall back, change precision, or return invalid results.
- A CPU fallback with a clear warning is acceptable. Successful execution alone does not establish numerical correctness or the requested execution backend.

## Workspace and coordination

Read applicable `AGENTS.md` instructions. Do the implementation in a separate Git worktree on a dedicated branch based on the current `gpu-bed-reader` development head. Keep this development checkout's existing edits and staged changes intact. Keep environments and large generated outputs outside the repository, following repository guidance.

Inspect `git worktree list` before starting. An existing numerical investigation uses branch `investigate/gpu-alpha-numerics`; inspect its current status and evidence without modifying its worktree. Coordinate changes rather than duplicating or overwriting that investigation. If sibling dependency changes are necessary, use separate worktrees for those repositories too. Do not publish packages, push release tags, or merge branches without explicit authorization.

The prompt and `specs/todo.md` may be uncommitted in the development checkout. Read them there if they are absent from the new worktree; do not stage or commit unrelated changes to transfer instructions.

## Inventory and behavior

Use `doc/source/index.rst` as the primary API checklist. Include:

- `single_snp`, with separate single-kernel and two-kernel paths.
- `single_snp_scale`.
- `single_snp_all_plus_select`.
- `single_snp_linreg`.
- `single_snp_select`, preserving its deprecated status and platform limitations.
- `epistasis`.
- `snp_set`.
- `FastLMM` and its public methods.
- `LinearRegression` and its public methods.
- `heritability_spatial_correction`.
- `compute_auto_pcs`, preserving its deprecated status and platform limitations.
- `manhattan_plot`.

Reconcile this list against exported entry points, including compatibility aliases. For each entry, record requested and actual backend/precision behavior, its relevant tests, and any limitations. Plotting and metadata operations need an appropriate input contract, not an artificial requirement for GPU computation. Document deliberate higher-precision statistics separately from an unsupported whole-workflow precision override.

Inspect existing implementation and tests before editing. Audit explicit arguments, `ARRAY_MODULE`, `ARRAY_DTYPE`, supplied arrays, cached models, and runner environments where applicable. A parent-process setting alone does not establish worker placement.

For unqualified numerical workflows, CPU/float64 is the recommended conservative baseline. First inspect existing documented behavior; preserve supported behavior and avoid unnecessary changes to algorithms or public signatures. Make every unsupported request visible with an informative warning identifying the affected API and actual fallback, or an informative error before unsupported work. Do not claim uniform float32 support merely because readers or precomputation use float32.

Use a small shared internal helper where it avoids duplicated policy. Restore scoped settings on success and failure, account for concurrent workflows and subprocesses, and preserve original exceptions. Default CPU/float64 calls should not gain irrelevant fallback warnings. Avoid repeated warnings from internal calls where a single public-boundary warning suffices. Missing GPU dependencies or native GPU support must not silently select CPU for single-kernel `single_snp`.

## Validation

- Test each documented entry point directly, independently of the notebook. Verify actual execution and warnings/errors, including settings restoration after success and failure. Include supported default behavior and applicable worker paths.
- Qualify all four single-kernel configurations across representative existing paths, including full/low rank, LOCO, cache use, covariates, and runners. Check array backend/dtype inside computation rather than inferring execution from CPU result tables.
- Check numerical accuracy, identities, ordering, result schema, finite masks, and edge cases against justified reference criteria. Add regression coverage for fixes. Preserve existing tolerances; never widen them just to pass.
- Investigate or incorporate the existing single-kernel fitted-h2 discrepancy and mouse float32 accuracy assessment. Preserve the earlier float32 epistasis failure as evidence; routing an unqualified request to CPU/float64 does not prove that the float32 algorithm was corrected.
- Run the narrow relevant tests, then the complete required validation appropriate to the changes. Preserve legacy test registration when adding tests. Report unavailable environments and unresolved failures explicitly.
- Rerun the full main notebook in all four configurations with fresh kernels as additional integration coverage. Preserve original source edits and saved outputs; save executed copies separately. Do not treat commented-out deprecated examples as API validation.

Useful prior evidence is linked from `specs/todo.md`, `specs/GPU_ALPHA_RELEASE_CHECKLIST.md`, `/mnt/c/deldir/fastlmm-notebook-validation-20261117/REPORT.md`, and `/mnt/c/deldir/fastlmm-mouse-four-way-20261118/REPORT.md`. These are historical evidence, not qualification of subsequent changes. The notebook's `num_threads` argument requires the newer PySnpTools runner, not published PySnpTools 1.1.0a1.

## Deliverables

Implement focused code and regression tests, document a support matrix linked from the API documentation and GPU guide, and update the worktree's todo status. Report exact tested revisions, environments, results, unavailable checks, and remaining compatibility or numerical risks. Include the worktree path, branch, and a concise suggested commit message. Do not declare the task complete while required API paths silently override requests or required validation remains unresolved.
