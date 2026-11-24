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

The strict `xp=None` rule (commit `8f37851`, Carl's decision November 23):
`xp=None` never copies an existing NumPy or CuPy array to the other module;
constructors, `.val` assignment, and reads of in-memory values raise
`ValueError`, while lists, `val=None`, and file-decoded values follow
`ARRAY_MODULE`.

## FaST-LMM

Comparison: `v1.0.0` (`279fdb01f6cfb8f4ff1a09d5c3322488546c4ed2`) to
`gpu-bed-reader` (`8f67daad275402c827b64a7ae07bfb5afde46653`), approximately
10,630 changed lines across 66 files.

| Batch | Files and approximate size | Review focus |
| --- | --- | --- |
| Numerical and GPU paths (~830 lines) | `fastlmm/association/epistasis.py`, `single_snp.py`; `fastlmm/util/_backend.py`, `_gpu_warnings.py`, `_precision.py` | Precision policy, NumPy/CuPy selection, warnings, fallback boundaries, worker propagation, and numerical changes. |
| Public API and integration (~180 lines) | `fastlmm/association/{heritability_spatial_correction.py,single_snp_all_plus_select.py,single_snp_linreg.py,single_snp_scale.py,single_snp_select.py,snp_set.py,varcomp_test.py}`; `fastlmm/inference/{__init__.py,fastlmm_predictor.py,glmm.py,linear_regression.py,lmm.py,lmm_cov.py}`; `fastlmm/util/{compute_auto_pcs.py,util.py}`, `fastlmm/util/matrix/mmultfile.py` | Compatibility of forwarded backend/dtype options and interactions with existing inference and association APIs. |
| Tests (~1,790 lines) | `fastlmm/association/tests/test_api_policy.py`, `test_array_dtype.py`, `test_backend_environment.py`, `test_backend_selection.py`, `test_gpu_covariates.py`, `test_gpu_precision_warning.py`, `test_gwas.py`, `test_single_snp.py`, `testepistasis.py`, `tests/test.py` | Numerical tolerances, explicit failures and warnings, backend behavior, worker execution, and regression coverage. |
| Public guides and release notes (~2,900 lines) | `README.md`, `RELEASE_NOTES.md`, `doc/source/api_backend_precision.rst`, `doc/source/gpu.rst`, `doc/source/index.rst`, `doc/ipynb/FaST-LMM.ipynb` | Supported configurations, platform limits, installation, precision examples, and consistency of the main notebook with implementation. |
| Experiment notebook evidence (~2,690 lines) | `specs/experiments/gpu_f32.ipynb`, `specs/experiments/gpu_fastest_path.ipynb` | Source provenance, saved outputs, numerical identity and ordering, warnings, and benchmark claims. The saved configurations require reruns against current development heads. |
| Specifications and qualification tools (~2,100 lines) | `AGENTS.md`, `CLAUDE.md`, `specs/ANNUAL_UPDATE_RUNBOOK.md`, `specs/BRANCH_INVENTORY_2026.md`, `specs/GPU_ALPHA_RELEASE_CHECKLIST.md`, `specs/GPU_API_POLICY_PROMPT.md` (deleted), `specs/GPU_API_POLICY_STATUS.md` (deleted), `specs/GPU_API_SPEC.md` (deleted), `specs/GPU_DOCS_REVIEW.md`, `specs/GPU_F32_SPEC.md`, `specs/GPU_NUMERICAL_CORRECTNESS.md`, `specs/SYNTHETIC_BENCHMARK.md`, `specs/todo.md`, `tools/annual-update/README.md`, `tools/annual-update/{compare-notebook-outputs.py,countdown.sh,fastlmm-notebooks.sh,notebook-env.sh,refresh-notebook-outputs.py,runcountdown.sh}`, `tools/synthetic-benchmark/{benchmark.py,driver_synthetic.py}` | Review branch-maintenance changes, historical evidence, current qualification status, and reproducibility tools. |
| CI and packaging (~150 lines) | `.github/dependabot.yml`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `pyproject.toml`, `uv.lock` | Dependency bounds, optional GPU extra, supported Python versions, and build/release workflows. |

The FaST-LMM comparison includes the three specifications deleted by this
cleanup because they were part of the reviewed head. The current tree retains
`GPU_DOCS_REVIEW.md`, `GPU_F32_SPEC.md`, `GPU_NUMERICAL_CORRECTNESS.md`, and
`GPU_ALPHA_RELEASE_CHECKLIST.md` as evidence or open qualification records;
their historical bed-reader GPU statements are identified as such.
