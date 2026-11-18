<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

# GPU alpha release checklist

All three coordinated alphas are published. This checklist records their
validation and the remaining platform limitations. No new design decision or
publication action is required from Carl now.

## Upload order

1. **Done:** added 20 CPU wheels to the existing bed-reader `1.3.0a1` release: Intel Mac,
   Apple silicon, Linux ARM64, and Windows ARM64, Python 3.11–3.15. These have
   distinct filenames and do not replace the published NVIDIA x86-64 wheels.
2. **Done:** published PySnpTools `1.1.0a1` (one wheel and one source archive).
3. **Done:** published hashes match all 22 files and
   both development lockfiles use PyPI. Fresh pip/uv installations with the
   published prerequisites passed; the full FaST-LMM canonical suite passed
   204 tests with no failures or skips. Refreshed CPU CI passed: 39 PySnpTools
   jobs and 41 FaST-LMM jobs. FaST-LMM's monthly-only freshness job was skipped
   by its existing schedule condition.
4. **Done:** Carl published FaST-LMM `1.1.0a1` on November 18, 2026. Both
   PyPI hashes match the qualified artifacts.

The first two upload sets remain under `~/.cache/gpu-alpha-20261117/ready`
as a record; do not re-upload them. FaST-LMM's final files are under
`ready/fastlmm`; acceptance and publication are complete. Do not re-upload them.

All final CI gates passed. The completed upload command is retained as a record:

```bash
cd /home/carlk/.cache/gpu-alpha-20261117
sha256sum -c fastlmm-SHA256SUMS
uv tool run --from twine==6.2.0 twine upload \
  --config-file ~/.pypirc --repository pypi \
  --username __token__ --non-interactive \
  ready/fastlmm/*.whl ready/fastlmm/*.tar.gz
```

**Done:** verified PyPI hashes and a fresh installation of all three published
alphas. CPU/GPU reads, kernels, and LOCO association passed for float32/float64
under WSL2 and native Windows; all 142 installed regression tests passed again
under WSL2. Default installs still select bed-reader 1.2.1, PySnpTools 1.0.1,
and FaST-LMM 1.0.0. No release tag was pushed and nothing was merged into main.

## Validation already completed

- bed-reader's initial 11 files are published; their PyPI hashes match the
  qualified artifacts. Default installs still choose stable `1.2.1`.
- The additional CPU wheels passed the relevant Python and installed-wheel CI
  checks. No GPU capability is promised for those platforms.
- PySnpTools: 755 full canonical tests passed; minimal feature configuration
  passed 467 tests with 18 expected optional-feature skips; 97 installed
  GPU/precision tests passed. Direct and source-derived wheels match exactly.
- Combined PySnpTools/FaST-LMM alpha matrix: CPU workflows passed on Python
  3.11–3.15 under WSL2 and native Windows; 142 installed integration tests
  passed on each platform/version with CuPy, Python 3.11–3.14.
- Declared direct dependency minimums passed CPU workflow checks on Python
  3.11 and 3.14. Strict docs and independently executed guide examples passed.
- FaST-LMM's clean source archive contains no development Git override and
  produces a wheel identical to the tested candidate.

## Gates and open issues

- **Done:** removed FaST-LMM's temporary Git source and refreshed `uv.lock`
  against published PyPI packages. The tag-release workflow rejects any
  remaining development source overrides.
- **Done:** refreshed PySnpTools CPU CI passed all 39 jobs after publication,
  including the seven Mac/ARM configurations previously blocked by missing
  bed-reader wheels. FaST-LMM CPU CI passed all 41 applicable jobs.
- bed-reader CI: 56 jobs passed. Rust all-features checks failed on Linux ARM
  (CubeCL HIP `Vec<u8>`/`Vec<i8>` mismatch), Windows ARM and Intel Mac (LLVM
  bundle download returns 404). Keep these failures visible; no lint or feature
  coverage was suppressed. These are separate from the passing Python wheels.
- Explicit completion records confirm 262 distribution/BGEN tests and 204
  broad FaST-LMM canonical tests passed. The final published-prerequisite
  installation passed all 142 installed integration tests as well.
- **Done:** native Linux GPU acceptance passed on November 18, 2026 using
  a rented Vast.ai RTX 3060, Ubuntu 24.04 x86-64, Python 3.11.17, NVIDIA driver
  610.57.04, CUDA toolkit 13.3, and CuPy 14.2. The environment reported CUDA
  runtime 13.2 and driver API 13.3. Installed all three alphas from PyPI; CPU/GPU
  reads, kernels, and LOCO association agreed for float32/float64, the installed
  artifact smoke test passed, and all 142 regression tests passed in 65.26
  seconds. No runtime code changes were needed. Logs and environment metadata
  are saved under `~/.cache/gpu-alpha-20261117/native-linux-vast/`. This validates
  one native Linux host and Python version; additional hardware reports remain
  useful. Initial Python GPU support is NVIDIA x86-64, Python 3.11–3.14. Python
  3.15 is CPU-only pending CuPy wheels. Linux GPU wheels require glibc 2.35+.
- CuPy's driver/toolkit prerequisites remain external; `[cupy]` selects the
  Python dependency chain. Pip-managed CUDA components are an optional CuPy
  setup. Consumer GPU float64 warnings explain speed before storage cost.
- The separate notebook evaluation belongs to the other chat. Preserve Carl's
  uncommitted `doc/ipynb/FaST-LMM.ipynb` edits; they are not part of these commits.

CI runs: [bed-reader](https://github.com/fastlmm/bed-reader/actions/runs/44231693545),
[original PySnpTools](https://github.com/fastlmm/PySnpTools/actions/runs/44232241233),
[refreshed PySnpTools](https://github.com/fastlmm/PySnpTools/actions/runs/44240946747),
[final FaST-LMM](https://github.com/fastlmm/FaST-LMM/actions/runs/44241263702).

## Post-alpha runner and notebook integration

- Integrated PySnpTools commit `050fcbf` into `cupy-sub-matrix` as `75652ce`.
  Retained the published `bed-reader>=1.3.0a1` bounds and verified `uv lock --check`.
  The new direct threadpoolctl dependency and its Python-version bounds remain.
- Ported the three source edits from FaST-LMM commit `d539ccc` into the primary
  notebook. Preserved all 35 cells, metadata, and saved outputs, including Carl's
  separate review edits. A backup is under
  `/mnt/c/deldir/fastlmm-gpu-alpha-sync-20261118/FaST-LMM-before-integration.ipynb`.
- Combined Python 3.13 runner/GPU tests: 157 passed, 44 subtests passed. Both
  runner doctests passed. Runner tests also passed on Python 3.11 with
  threadpoolctl 3.5.0 and Python 3.15 with threadpoolctl 3.8.0: 15 tests and
  44 subtests on each. Ruff and notebook syntax checks passed. The two notebook
  NumPy scopes were executed under a CuPy/float32 default and restored it.
- The full notebook was not rerun during integration; preserved outputs are
  earlier evaluation results. The older worktree benchmark is not a new
  benchmark of this combined state.
- TODO0 Before the next alpha, release the updated PySnpTools runner and qualify
  the notebook against that published package. Published 1.1.0a1 has no
  `num_threads` keyword; the revised notebook requires the development checkout
  until the next alpha. Do not republish the existing alpha version.

## Completed development-source cleanup

The original reminder is retained here after removing its configuration:

```toml
# TODO0 After PySnpTools 1.1.0a1 is published, remove this development-only
# source and refresh uv.lock against PyPI before publishing FaST-LMM's alpha.
[tool.uv.sources]
pysnptools = { git = "https://github.com/fastlmm/PySnpTools", rev = "5ff6a0ee95c3881fef1d77a7e89a0a0da4313f78" }
```

Completed after PySnpTools 1.1.0a1 publication (may no longer apply).

## Post-alpha numerical investigation

The [integrated numerical record](GPU_NUMERICAL_CORRECTNESS.md) reproduces all
three earlier findings against the development heads. The epistasis product
rounding correction has a regression that fails before the fix. Follow-up
investigation found 11 additional legacy float32 fixture failures, so epistasis
now explicitly requires CPU float64 computation, warns on other requested
defaults, enforces worker precision, and rejects float32 eigen caches. This
qualifies CPU float64, not native float32; the original failures remain evidence.

Independent float64 h2 variation is explained by likelihood rounding. A
long-double derivative reference bounds the observed parameter error below
`4.982e-8` and maximum relative P error at about `0.000031%`. Recommendation:
retain the optimizer and document the precision limitation. The original
strict diagnostic exceedances remain recorded; no tolerance or expected output
was changed and no additional human approval is implied.

On November 18, 2026, Carl accepted the mouse GPU32 precision tradeoff for the
alpha with a warning and concrete P-value example in the guide and dtype docs.
This is an approved precision exception, not a passing strict comparison or
completion of Carl's broader review. Full notebook runs, CPU thread benchmarks,
installed-artifact acceptance, human review, and the next release remain open.

## Canonical combined development state

The API policy and numerical changes are combined in `investigate/gpu-alpha-numerics`.
[Combined status](GPU_API_POLICY_STATUS.md) records the passing 227-case canonical
run, separate completion of its five HDF5 skips, 60 combined regression tests,
Python boundary checks, and exact source/evidence hashes. No tolerance or
expected output was widened. This does not qualify a new published artifact.
The stopped policy chat's four completed notebook executions are preserved;
their numerical cross-comparison and final-source reconciliation remain next.
CPU thread benchmarks, Carl's review, platform/artifact qualification, and
release preparation remain open. Nothing was published or merged into main.
