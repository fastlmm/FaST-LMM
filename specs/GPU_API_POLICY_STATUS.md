<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

# Canonical combined API policy and numerical change

The canonical development worktree is
`/home/carlk/.cache/fastlmm-correctness-20261118/FaST-LMM`, branch
`investigate/gpu-alpha-numerics`, based on FaST-LMM `d37dccf845dc6da35b77c514360514ed17cafc9f`.
It combines the numerical investigation with the uncommitted changes from
`/home/carlk/programs/InstallTest/fastlmm-api-policy` (`implement/gpu-api-policy`).
Both edit sets and subsequent source updates are preserved under
`/mnt/c/deldir/fastlmm-correctness-20261118/integration/`. The source policy
worktree and primary checkout were not modified by this integration.
The source worktree's test-induced executable mode change was not imported.

PySnpTools development imports use `75652cec21624db4d5fc7720cca94dd162f706b2`.
The bed-reader overlay's Python runtime matches `71f62e4414e770867a33ae8dfa974bd26f66fc55`;
its identified development CUDA binary, source equivalence, and hash are in the
[numerical evidence report](/mnt/c/deldir/fastlmm-correctness-20261118/REPORT.md).
These are development-source checks, not published-artifact acceptance.

## Final execution contract

- Standalone single-kernel `single_snp` honors NumPy/CuPy and float32/float64,
  including explicit arguments, full/low rank, LOCO, cached models, and workers.
  Missing GPU dependencies fail explicitly. The accepted mouse float32 example
  and the independently fitted-h2 roundoff limitation remain documented.
- Two-kernel and other unqualified numerical APIs use CPU float64 with a warning
  naming the API, requested configuration, and actual fallback. Supplied device
  or float32 data are copied with a warning; existing rounding cannot be undone.
  Invalid settings and unqualified cached model/eigen arrays raise useful errors.
- The shared workflow lock restores scoped settings on success and failure.
  A context-local CPU64 requirement follows nested calls without leaking into
  concurrent or subsequent standalone single-kernel workflows.
- Serialized runners normalize generator setup/advancement, task bodies, nested
  work trees, and reduction. Scale matrix helpers also establish explicit CPU64
  boundaries. This prevents runner task-environment overrides from silently
  reintroducing GPU/float32 into CPU-only workflows.
- Spatial work items retain tuple behavior locally and use the existing
  cloudpickle dependency for standard multiprocessing transport of reader
  objects carrying array-module references. Worker computation is scoped too.
- Plotting accepts host data at its existing precision and rejects device arrays
  with transfer instructions. Deprecated API/native-platform limitations remain.

The complete public-entry-point inventory is in
[api_backend_precision.rst](../doc/source/api_backend_precision.rst).

## Integration corrections and evidence

The initial combined regression exposed missing worker boundaries for deprecated
selection's final single-kernel call and scale matrix reduction, plus standard
multiprocessing serialization of readers containing a NumPy module. The failed
run is retained in `integration/combined-policy-tests.log`; the focused worker
regression passes after the corrections. Additional tests cover missing CuPy
for a CPU-only fallback, simultaneous CPU-only and standalone workflows, exact
GPU-input numerical comparisons, and the same-model effect-size comparison.
No tolerance or expected fixture was widened.

Before API-policy integration, the numerical canonical suite ran 209 tests:
204 passed and five optional HDF5 tests skipped because h5py was absent. Those
five tests passed separately in an existing complete environment. The original
GPU-read effect-size test failed on the untouched development code; keeping its
independent-fit P-value check while comparing effect sizes at the same h2 fixes
that model mismatch without changing its tolerance. Details and original
failure logs are retained under `followup/`.

Final combined validation passes:

- Canonical suite: 227 tests run, 222 passed and five HDF5 skips in the core
  environment. The five skipped tests all pass in the complete HDF5 environment
  against the same final source snapshot. No tests were disabled or weakened.
- Python 3.13 combined API/epistasis/backend/covariate regressions: 60 passed,
  94 subtests passed, actual GPU exercised.
- Python 3.11 API/epistasis: 37 passed, 94 subtests passed.
- Python 3.15 API/epistasis: 35 passed, 86 subtests passed; two expected GPU skips
  (`test_single_kernel_cupy_matrix` and `test_gpu_input_and_plot_contract`).
  CPU-only fallbacks for CuPy requests still run and pass without CuPy.
- Ruff passes every changed Python file. Strict Sphinx build passes with actual
  canonical imports asserted and rendered API warning/example checked.
- Fresh combined-tree numerical controls reproduce the original synthetic and
  mouse results exactly at the recorded maxima. Independent float64 comparisons
  remain diagnosed fitting-roundoff limitations; shared-h2 controls reach
  `2.49e-14` synthetic and `2.13e-13` mouse in log10(P). Mouse GPU32 maximum
  `3.7262e-4` remains the accepted precision example. Epistasis returns exactly
  the CPU64 reference under all four requested defaults, with a warning on
  unsupported requests; native float32 epistasis is not qualified.

The independent product regression still fails when only product promotion is
removed while the CPU64 fallback is retained. All original failure logs and
strict tolerances remain preserved. Exact commands, environments, hashes,
snapshots, and results are under
`/mnt/c/deldir/fastlmm-correctness-20261118/integration/`.

The stopped source-policy chat also completed four notebook executions under
`/mnt/c/deldir/fastlmm-api-policy-20261118/notebooks-final/`: `numpy-float64-final`,
`cupy-float64`, `numpy-float32`, and `cupy-float32`. Each recorded 18 successful
execution/capture steps. Their numerical cross-comparison and source-manifest
reconciliation remain subsequent qualification work. They are not labeled as
validation of this final combined source. The handoff is preserved as
`integration/final-policy-handoff.md`; do not rerun the notebooks merely because
that work was done in a separate worktree.

## Remaining work

Full main-notebook execution in all four configurations, CPU thread benchmarks,
Carl's cross-repository review, installed-artifact/platform qualification, and
release preparation remain in [todo.md](todo.md). No publication, release tag,
or merge into main is authorized. Preserved notebook outputs are historical;
this integration does not claim a fresh full-notebook or performance result.
