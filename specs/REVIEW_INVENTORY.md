<!-- TODO0 consider deleting this inventory after Carl's review is complete. -->

# GPU alpha review inventory

This inventory supports Carl's review of the current GPU changes in PySnpTools
and FaST-LMM and the CPU-only migration in bed-reader. Each base is the
comparison point requested for the review; changed-line estimates count both
additions and deletions and are rounded to the nearest ten. The listed head
commit identifies the reviewed snapshot. Review later changes separately.

## bed-reader

Comparison: `origin/main` (`a462625e4c44a9c0b18703501559f219bd2cd63c`) to
`implement/cpu-only` (`b535ef30b239adcb434065acc57385c93c34c679`), approximately
1,040 changed lines across 23 files.

| Batch | Files and approximate size | Review focus |
| --- | --- | --- |
| CPU reader and public API (~440 lines) | `bed_reader/_array_module.py`, `bed_reader/_open_bed.py`, `src/bed_cloud.rs`, `src/lib.rs`, `src/python_module.rs` | Shared index resolution, axis handling, dtype defaults, cloud-reader parity, and the removal of GPU-only bindings. |
| Regression tests (~210 lines) | `bed_reader/tests/test_array_dtype.py`, `bed_reader/tests/test_open_bed.py`, `src/tests.rs`, `tests/tests_api.rs`, `tests/tests_api_cloud.rs` | Index edge cases, dtype policy, and CPU API behavior. |
| Documentation and release notes (~120 lines) | `CHANGELOG.md`, `README-rust.md`, `README.md`, `doc/source/conf.py`, `doc/source/index.rst`, `src/supplemental_documents/options_etc.md` | CPU-only release wording, intersphinx retry behavior, and user-facing API accuracy. |
| CI and packaging (~260 lines) | `.cargo/config.toml`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `Cargo.lock`, `Cargo.toml`, `pyproject.toml`, `uv.lock` | MSRV-aware resolution, removed GPU build/release paths, dependency features, and artifact workflows. |

## PySnpTools

Comparison: `v1.0.1` (`bd0a04244e4c0bdef0129f757375b84df4e6b062`) to
`cupy-sub-matrix` (`d839fe2a76fbd9d6cda6c8ee6194cc4867967b62`), approximately
3,940 changed lines across 53 files.

| Batch | Files and approximate size | Review focus |
| --- | --- | --- |
| Reader and public API (~1,150 lines) | `pysnptools/distreader/bgen.py`, `distdata.py`, `distreader.py`; `pysnptools/kernelreader/_subset.py`, `kerneldata.py`, `kernelreader.py`, `snpkernel.py`; `pysnptools/pstreader/pstdata.py`, `pstreader.py`; `pysnptools/snpreader/_subset.py`, `bed.py`, `dat.py`, `dense.py`, `ped.py`, `pheno.py`, `snpdata.py`, `snpreader.py`; `pysnptools/util/__init__.py`, `_array.py`, `gpu.py`; `pysnptools/util/mapreduce1/runner/_threads.py`, `localinparts.py`, `localmultiproc.py` | Backend selection and propagation, index/view behavior, BGEN compatibility, thread control, and public API boundaries. |
| Numerical and GPU paths (~330 lines) | `pysnptools/standardizer/_cupy.py`, `beta.py`, `betatrained.py`, `diag_K_to_N.py`, `standardizer.py`, `unit.py`, `unittrained.py`; `pysnptools/util/_gpu_warnings.py`, `_precision.py` | CuPy allocation and standardization, dtype defaults, precision warnings, and trained-statistic behavior. |
| Tests (~1,470 lines) | `pysnptools/kernelreader/test.py`, `pysnptools/test.py`, `pysnptools/test_array_dtype.py`, `test_gpu_allocator.py`, `test_gpu_api.py`, `test_gpu_precision_warning.py`, `test_gpu_standardize_general.py`, `test_gpu_support.py`, `pysnptools/util/mapreduce1/test_threads.py`, `pysnptools/util/test.py` | CPU/GPU parity, ownership, dtype and layout, optional dependency errors, runner limits, and regression coverage. |
| User documentation and notebook (~850 lines) | `CHANGELOG.md`, `README.md`, `doc/ipynb/gpu_workflow.ipynb`, `doc/source/gpu-api-preview.rst`, `doc/source/gpu.rst`, `doc/source/index.rst`, `specs/GPU_DOCUMENTATION_AUDIT.md` | CPU-read/CuPy-transfer workflow, public support matrix, installation requirements, warnings, and saved example outputs. |
| CI and packaging (~140 lines) | `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `pyproject.toml`, `uv.lock` | Optional GPU dependency, build metadata, supported Python versions, and release validation. |

### Carl's PySnpTools review against `main` (started November 23, 2026)

Comparison: `git diff origin/main...cupy-sub-matrix` (merge base `8c68afc`),
about 4,700 changed lines across 64 files at `9ae0a1d`. This supersedes the
batch table above for tracking Carl's progress.

| Batch | Files | Status |
| --- | --- | --- |
| 1. GPU and precision core | `util/_array.py`, `util/_precision.py`, `util/_gpu_warnings.py`, `util/gpu.py`, `util/_device.py`, `standardizer/_cupy.py`, `standardizer/standardizer.py` | Approved November 23 at `69112e4`; delta (`_device.py`, the explicit stats-shape check) approved at `9ae0a1d`. Review fixed constant fractional SNPs standardizing to +/-1 on CuPy and the Python CPU path. |
| 2. Standardizers | `standardizer/{unit,unittrained,beta,betatrained,diag_K_to_N}.py` | Approved November 23 at `077f35b`. Review fixed `UnitTrained` reordering with repeated SNP IDs (silently used the last copy's moments) and the Python Beta path accepting means outside [0, 2]. |
| 3. Array utilities and containers | `util/__init__.py`, `pstreader/{pstdata,pstreader,_oneshot,_subset,pstmemmap}.py`, `snpreader/_subset.py`, `kernelreader/_subset.py` | Approved November 23 at `30ebc30`. Review removed a dead CuPy-fallback flag and corrected the `sub_matrix` dtype default documentation. |
| 4. SNP readers | `snpreader/{__init__,snpreader,snpdata,snpmemmap,bed,dat,dense,ped,pheno}.py`, `util/generate.py` | Approved November 23 at `bba4f6b`. Review fixed the SnpData Identity kernel fast path ignoring an explicit `block_size`, made `SnpData.standardize(block_size=...)` warn like the standardizers (FaST-LMM `11833d2` stopped passing it), and noted the explicit-block-size rounding change in the changelog. |
| 5. Kernels and distributed readers | `kernelreader/{kerneldata,kernelreader,snpkernel}.py`, `distreader/{bgen,distdata,distreader,distmemmap}.py` | Approved November 23 at `89db1f1`. Review added explicit NumPy copies in every writer (writing is an explicit request to leave the GPU), normalized deprecated writer argument order first, kept F layout in `asnumpy`, and fixed the `Dense.write` reversed-argument typo. |
| 6. Thread control | `util/mapreduce1/runner/{_threads,localinparts,localmultiproc,localmultithread}.py` | Approved November 23 at `d48b746`. Review made an unset runner `num_threads` raise AttributeError rather than KeyError. |
| 7. Tests | `test*.py` | Approved November 23 at `d200e3b`. Agent review found that CI's legacy runner did not collect the pytest-only modules; `80fac2d` added an explicit pytest CI step for them. |
| 8. Docs and packaging | `CHANGELOG.md`, `README.md`, `doc/`, `pyproject.toml`, `uv.lock`, `.github/`, `specs/` | Approved November 23 at `4645f66`. Review corrected release notes and GPU documentation, extended CPU pytest coverage to Python 3.15 and Windows ARM, added installed-artifact Python 3.15 tests, and removed two completed specifications. |

A post-approval review found that blocked `DiagKtoN` kernels fitted a separate
normalization factor to every SNP block, an old bug newly exposed for narrow
data by the explicit-`block_size` change. PySnpTools `9d6f294` scales the summed
kernel once, returns `DiagKtoNTrained`, adds CPU/GPU regressions, and rejects
invalid manually supplied `UnitTrained` deviations before mutation. This
supersedes the previously reviewed `8a9c624` head. Corrected head `0510bfc`
passed all 39 CI jobs (run 45417729712) and final source/GPU/artifact
qualification on November 24. The subsequent release-record refresh changes
documentation only and still requires its own final CI result.

The strict `xp=None` rule (commit `8f37851`, Carl's decision November 23):
`xp=None` never copies an existing NumPy or CuPy array to the other module;
constructors, `.val` assignment, and reads of in-memory values raise
`ValueError`, while lists, `val=None`, and file-decoded values follow
`ARRAY_MODULE`.

## FaST-LMM

Comparison: `git diff v1.0.0...gpu-bed-reader` (merge base `9da663a`) at
`4bad15d`, approximately 11,930 changed lines across 77 files. Notebook JSON
and generated numerical evidence inflate the textual line counts; review those
batches semantically as noted below.

| Batch | Files and approximate size | Review focus | Status |
| --- | --- | --- | --- |
| 1. `single_snp` numerical and GPU path (~610 lines) | `fastlmm/association/single_snp.py` | Precision and dtype policy, NumPy/CuPy switching, covariate placement, worker propagation, memory behavior, and result changes. | Approved November 24, 2026 at `8a98823`. Review fixed integer `h2` corrupting the cache's NaN sentinel (`3e6fff3`) and float32 validation accepting infinity (`6715e09`). Validation added usable-CUDA checks (`cd0078f`, `ffb4e33`). Later `single_snp` changes are tracked with batch 2. |
| 2. Remaining numerical and GPU paths (~430 lines) | `fastlmm/association/epistasis.py`; `fastlmm/inference/lmm_cov.py`; `fastlmm/util/{_backend,_gpu_warnings,_precision}.py` | Numerical equivalence, GPU linear algebra, fallback boundaries and warnings, thread/environment propagation, and float32 behavior. | Approved November 24, 2026 at `d035567`. Review fixed the `LocalMultiThread` deadlock (`66390cd`) and preserved small float32 residual signals while requiring affected older caches to be rebuilt (`8148066`). |
| 3. Public API and integration (~150 lines) | `fastlmm/association/{heritability_spatial_correction,single_snp_all_plus_select,single_snp_linreg,single_snp_scale,single_snp_select,snp_set,varcomp_test}.py`; `fastlmm/inference/{__init__,fastlmm_predictor,glmm,linear_regression,lmm}.py`; `fastlmm/util/{compute_auto_pcs,run_fastlmmc,util}.py`; `fastlmm/util/matrix/mmultfile.py` | Compatibility of forwarded backend/dtype options and interactions with existing inference and association APIs. | Approved November 24, 2026 at `e826bff` with fixes for threaded spatial correction and inherited predictor state validation. The pre-existing `fastlmmc` SIGILL remains tracked in `specs/todo.md`. |
| 4. API-policy tests (~770 lines including Batch 3 regression tests) | `fastlmm/association/tests/test_api_policy.py` | Public defaults, fallback warnings, worker propagation, nested execution, and failure behavior. | Approved November 24, 2026 at `485b1fc`. The cache matrix verifies cache loads and forbids recomputation. The failure-restoration test requires the exact injected exception. Both tests rejected the simulated regressions. |
| 5. Backend and numerical tests (former 5–6) | Backend/dtype/GPU modules plus `test_gwas.py`, `test_single_snp.py`, and `testepistasis.py` | Check the regression contracts and numerical tolerances. | Approved November 24, 2026 at `4bad15d`. The dtype race and warning contract were tightened in `4c797a7`. The published-prerequisite suite had 135 passes and 26 CUDA skips before the LOCO expansion. The expanded covariate module had six passes and six CUDA skips. |
| 6. User documentation and notebook evidence (former 7–8, 10–12) | README, release notes, Sphinx guides, notebooks, and qualification specs | Review the current installation instructions, cache migration, and what the notebook evidence supports. | Approved November 24, 2026. `9337de9` corrects the alpha and notebook claims; `06adad5` fixes the two first-party association doctests. The re-review fixed the notebook's local GPU-guide link and the README's branch link and added the `fastlmmc` release note. A fresh notebook run completed with no errors. The three vendored PCA doctests remain open in `specs/todo.md`. |
| 7. CI, artifacts, and maintenance tools (former 9, 13–14) | Workflow and dependency changes, artifact contents, runner helpers, and generated feature-selection evidence | Review the CI test step, dependency bounds, package contents, and whether to untrack the historical CSVs. | Approved November 24, 2026. `da89be7` wires pytest into CI; `34b9de3`, `7afea48`, and `2ebb688` isolate output and harden qualification helpers. Locally built wheel and source archive passed installed smoke tests. Carl accepted the CI pytest scope with CUDA qualified separately, the prerelease dependency bounds, the release workflow guards, and Dependabot's monthly `uv` group. The five `tmp/feature_selection/` files are untracked and ignored. |

### Consolidated review commands

Use the committed branch diff for the baseline and inspect any pending edits
to the same paths with `git diff -- <paths>`.

**5. Backend and numerical tests**

```bash
git diff v1.0.0...gpu-bed-reader -- \
  fastlmm/association/tests/{test_array_dtype,test_backend_environment,test_backend_selection,test_gpu_covariates,test_gpu_precision_warning,test_gwas,test_single_snp,testepistasis}.py
```

On Linux/WSL2 with Python 3.14.5, published PySnpTools 1.1.0b1 and
bed-reader 1.3.0b3, the canonical runner passed 230 tests and skipped six
CUDA cases. The focused suite passed 135 tests and 12 subtests, with 26
CUDA skips. The five CI pytest modules also passed with CuPy imports
disabled (18 passed, 30 skipped).

**6. User documentation and notebook evidence**

```bash
git diff v1.0.0...gpu-bed-reader -- README.md RELEASE_NOTES.md RELEASING.md \
  doc/source doc/ipynb/FaST-LMM.ipynb tests/notebooks.toml \
  fastlmm/association/{single_snp_scale,snp_set}.py \
  fastlmm/association/tests/test_single_snp_scale.py \
  specs AGENTS.md CLAUDE.md
```

Historical notebooks and timing records were reviewed for provenance and
consistency. Their timings still describe the original configurations.
The main notebook's interrupted output requires a full rerun. The
strict Sphinx build passed, and both fixed association doctests passed
under pytest and their canonical unittest wrappers. The broader doctest
rerun had 14 passes and three known failures in the vendored PCA examples,
which remain tracked in `specs/todo.md`.

**7. CI, artifacts, and maintenance tools**

```bash
git diff v1.0.0...gpu-bed-reader -- .github conftest.py pyproject.toml uv.lock tests/test.py \
  external/fastlmmc/fastlmmc fastlmm/association/Fastlmm_autoselect/fastlmmc \
  tools fastlmm/feature_selection/test.py tmp/feature_selection
git diff -- pyproject.toml uv.lock
```

The two executable blobs match the baseline and gained executable permissions. After the review (November 24, 2026), both bundled copies were replaced with `fastlmmc` 2.08 (Apache 2.0, oneMKL 2026.2.1) from the verified v2.08 release archives, with its license, notice, and third-party licenses beside the executables.
The reviewed wheel and source archive preserve the packaged executable bit and
exclude the generated feature-selection files. The reviewed dependency-bound
and lockfile edits are in `5318980`. The artifacts built with those edits
still have the existing `1.1.0a1` version and are local test
artifacts. Platform CI and GPU acceptance remain open.
