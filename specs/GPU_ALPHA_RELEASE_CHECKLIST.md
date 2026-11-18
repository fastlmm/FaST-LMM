<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

# GPU alpha release checklist

This checklist records the remaining decisions and actions. No new design
decision is required from Carl now. Publication remains a maintainer action.

## Upload order

1. **Done:** added 20 CPU wheels to the existing bed-reader `1.3.0a1` release: Intel Mac,
   Apple silicon, Linux ARM64, and Windows ARM64, Python 3.11–3.15. These have
   distinct filenames and do not replace the published NVIDIA x86-64 wheels.
2. **Done:** published PySnpTools `1.1.0a1` (one wheel and one source archive).
3. **In progress:** published hashes match all 22 files and both development
   lockfiles now use PyPI. Rerun PySnpTools CI and finish FaST-LMM acceptance.
4. Publish FaST-LMM `1.1.0a1` after its final published-dependency check.

The first two upload sets remain under `~/.cache/gpu-alpha-20261117/ready`
as a record; do not re-upload them. FaST-LMM's final files are under
`ready/fastlmm`; their publication is still pending acceptance.

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
- Previous PySnpTools CI: 31 jobs passed; seven Mac/ARM binary-only jobs failed
  because the additional bed-reader wheels were not published yet. A refreshed
  run follows publication; record its final result before closing this item.
- bed-reader CI: 56 jobs passed. Rust all-features checks failed on Linux ARM
  (CubeCL HIP `Vec<u8>`/`Vec<i8>` mismatch), Windows ARM and Intel Mac (LLVM
  bundle download returns 404). Keep these failures visible; no lint or feature
  coverage was suppressed. These are separate from the passing Python wheels.
- The separate distribution/BGEN suite now has an explicit completion record:
  262 tests passed. The broad FaST-LMM canonical suite is running again with
  per-test progress and a completion record; do not count it as passed early.
- Native Linux GPU execution remains unverified; local Linux GPU checks use
  WSL2. Initial Python GPU support is NVIDIA x86-64, Python 3.11–3.14. Python
  3.15 is CPU-only pending CuPy wheels. Linux GPU wheels require glibc 2.35+.
- CuPy's driver/toolkit prerequisites remain external; `[cupy]` selects the
  Python dependency chain. Pip-managed CUDA components are an optional CuPy
  setup. Consumer GPU float64 warnings explain speed before storage cost.
- The separate notebook evaluation belongs to the other chat. Preserve Carl's
  uncommitted `doc/ipynb/FaST-LMM.ipynb` edits; they are not part of these commits.

CI runs: [bed-reader](https://github.com/fastlmm/bed-reader/actions/runs/44231693545),
[PySnpTools](https://github.com/fastlmm/PySnpTools/actions/runs/44232241233).

## Completed development-source cleanup

The original reminder is retained here after removing its configuration:

```toml
# TODO0 After PySnpTools 1.1.0a1 is published, remove this development-only
# source and refresh uv.lock against PyPI before publishing FaST-LMM's alpha.
[tool.uv.sources]
pysnptools = { git = "https://github.com/fastlmm/PySnpTools", rev = "5ff6a0ee95c3881fef1d77a7e89a0a0da4313f78" }
```

Completed after PySnpTools 1.1.0a1 publication (may no longer apply).
