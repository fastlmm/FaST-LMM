# Python 3.14 Support and Toolchain Modernization Specification

<!-- todo0 consider deleting this spec once the work below is implemented and released. -->

## Status

Proposed.

## Objective

Add complete Python 3.14 support to FaST-LMM while preserving the same supported
Python range as `bed-reader`: Python 3.10, 3.11, 3.12, 3.13, and 3.14.

Because FaST-LMM's current toolchain and CI were last substantially updated in
2024, this work also modernizes dependency management, packaging metadata,
testing, CI, and release automation. Support includes dependency resolution,
installation, the full automated test suite, source-distribution and wheel
building, clean artifact installation, documentation, and release notes on all
supported operating systems and architectures.

## Guiding Model

Use the current `bed-reader` project as the model for:

- Python 3.10 through 3.14 support.
- Python 3.14's NumPy requirement.
- `uv` for Python installation, dependency resolution, environments, and
  command execution.
- Separate Intel and Apple Silicon macOS runners.
- Testing with both minimal and complete optional dependencies where relevant.

Do not copy `bed-reader` mechanically. FaST-LMM should use current `uv` and
GitHub Actions practices even where `bed-reader` retains older patterns such as
manual virtual-environment activation, an outdated `setup-uv` action, or
uncommitted lockfiles.

`uv` is the Rust-based Python project and package manager used for the project
workflow and build backend. Its managed Python installations come from
Astral's `python-build-standalone` distributions. FaST-LMM will migrate from
setuptools to `uv_build`; it will not adopt maturin or rewrite Python or native
dependencies in Rust.

## Current State

- `pyproject.toml` requires Python 3.10 or newer and advertises Python 3.10
  through 3.13.
- `.github/workflows/ci.yml` tests Python 3.10 through 3.13 on
  `ubuntu-latest`, `windows-2022`, `macos-13`, and `macos-14`.
- CI uses `astral-sh/setup-uv@v3`, runs `uv python install`, manually activates
  `.venv`, and permits dependency prereleases.
- CI runs lint and package builds redundantly in every OS/Python matrix entry.
- CI directly runs `tests/test.py`; it has not demonstrated that pytest
  discovery and the configured doctests are all included.
- `uv.lock` is ignored rather than committed, so ordinary CI is not frozen or
  reproducible.
- NumPy and SciPy are direct runtime imports, but neither is declared as a
  direct runtime dependency.
- NumPy, SciPy, Cython, and wheel are listed as build requirements even though
  no root-package extension build or Cython source has been identified.
- The package uses setuptools with explicit package and package-data lists that
  must be translated and verified during migration to `uv_build`.
- Development dependencies are split between a published `dev` extra and the
  legacy `[tool.uv].dev-dependencies` field.
- The package uses the deprecated table form of the `license` field.
- The repository has no automated, trusted-publishing release workflow.
- FaST-LMM depends on the maintainer-owned sibling projects
  `pysnptools>=0.5.14` and `fastlmmclib>=0.0.7`.

## Required Prerequisite Releases

The maintainer must update, test, and release **both `pysnptools` and
`fastlmmclib` with Python 3.14 support** before completing the FaST-LMM release.
These projects must not be treated as optional or replaced with unpublished
local checkouts in final verification.

For each prerequisite project:

1. Match `bed-reader`'s supported range of Python 3.10 through 3.14.
2. Adopt the applicable toolchain, metadata, CI, artifact-testing, and release
   practices in this spec.
3. Add Python 3.14 package metadata and CI coverage.
4. Apply the Python 3.14 NumPy requirement used by `bed-reader` where the
   project directly depends on NumPy.
5. Build all distributed native artifacts for Python 3.14 and all supported
   platforms and architectures.
6. Test installation from the built artifacts in clean Python 3.14
   environments.
7. Publish a release containing those artifacts.
8. Update FaST-LMM's lower bounds for `pysnptools` and `fastlmmclib` to the
   first released versions that provide Python 3.14 support.
9. Create a matching version tag that points to the exact source commit used
   to build the published artifacts.

FaST-LMM CI may temporarily test unreleased prerequisite commits or artifacts
while coordinating the work. Release acceptance must test the published
packages from the package index used by end users.

### Dependency-ordered project work

Perform and release the work in this order:

1. **fastlmmclib**
   - Regenerate the Cython-generated extension sources with a current,
     Python-3.14-compatible Cython release rather than relying on the old
     checked-in generated C/C++ as-is.
   - Build and test the native wheels for every supported operating system,
     architecture, and Python version, including clean Python 3.14 artifact
     installation.
   - Publish the tested artifacts from a matching version tag.
2. **PySnpTools**
   - Fix the broken tutorial-data references reported in
     [PySnpTools issue #10](https://github.com/fastlmm/PySnpTools/issues/10).
     The `pysnptools.hashdown.json` manifest must not refer to unavailable
     `doc/ipynb/all.bed`, `.bim`, or `.fam` objects.
   - Add a focused integration smoke test that exercises `example_file(...)`
     against the tutorial dataset and verifies that each required file can be
     retrieved and passes its declared integrity check. Keep network-dependent
     validation in an appropriate integration or release-gate job rather than
     making ordinary offline unit tests flaky.
   - Complete the Python 3.10-through-3.14 matrix and artifact tests, then
     publish the tested artifacts from a matching version tag.
3. **FaST-LMM**
   - Consume the released prerequisite versions from the end-user package
     index for final tests; do not qualify the release against sibling source
     checkouts.
   - Complete the SciPy/statsmodels investigation and Matplotlib regression
     coverage described below.
   - Make and document the Python 3.14 BGEN support decision described below.
   - Publish the exact tested artifacts from a matching version tag only after
     both prerequisite releases are available.

## `uv` Project Workflow

Make `uv` the standard interface for Python installation, dependency
resolution, environment synchronization, running tools and tests, building,
and publishing.

### Versioning and installation

- Upgrade from `astral-sh/setup-uv@v3` to the current reviewed release. At the
  time of this toolchain review, upstream documented `setup-uv` 8.1.0; recheck
  at implementation time.
- Pin the action to a full commit SHA and include the human-readable release in
  a comment.
- Pin the `uv` tool version explicitly rather than accepting an unreviewed
  latest version.
- Use the action's `python-version` input to select the matrix interpreter.
  Do not retain a redundant `uv python install` step unless a verified platform
  limitation requires it.
- Enable `setup-uv`'s built-in cache and let it key from the project metadata
  and lockfile.
- Use `uv run` for commands. Do not manually activate `.venv`, including on
  Windows.

The intended pattern is:

```yaml
- uses: astral-sh/setup-uv@<full-commit-sha> # vX.Y.Z
  with:
    version: "<reviewed-uv-version>"
    python-version: ${{ matrix.python-version }}
    enable-cache: true

- run: uv sync --frozen --all-extras --all-groups
- run: uv run --frozen python tests/test.py
```

Exact reviewed versions and SHAs belong in the implementation change and must
be maintained through dependency-update pull requests.

### Lockfile policy

- Remove `uv.lock` from `.gitignore`, generate it for the complete supported
  Python range, and commit it.
- Verify that the lockfile contains valid resolutions for Python 3.10 through
  3.14 and all supported platforms, not only the developer's interpreter.
- Ordinary pull-request CI must use `uv sync --frozen`; a stale lockfile is a
  failure.
- Commands run within the project environment should use `uv run --frozen`.
- The lockfile governs development and CI reproducibility. Published metadata
  remains governed by `[project.dependencies]` and must not pin end users to
  the lockfile.

### Stable, minimum, and future dependency testing

Use three distinct resolution policies:

1. **Required locked CI:** use the committed lockfile and stable releases.
2. **Required lower-bound CI:** resolve with `--resolution lowest-direct` and
   test that the declared direct-dependency lower bounds are truthful. Run at
   least on Python 3.10; add Python 3.14 when version markers create different
   lower bounds.
3. **Scheduled freshness CI:** run a new highest-resolution stable dependency
   solve, such as `uv lock --upgrade` in the disposable CI checkout, then sync
   and test. This detects ecosystem drift that frozen CI cannot see.

Do not use `--prerelease allow` in required CI or final release verification.
If prerelease ecosystem testing is useful, place it in a clearly labeled,
non-blocking scheduled job separate from release qualification.

## Dependency and Build-System Requirements

### NumPy

Follow `bed-reader`'s Python 3.14 runtime dependency policy:

```toml
"numpy>=1.22.0; python_version < '3.14'",
"numpy>=2.3.5; python_version >= '3.14'",
```

Add these entries to `[project.dependencies]` because FaST-LMM imports NumPy
directly. Do not raise the NumPy minimum for Python 3.10 through 3.13 merely to
support Python 3.14.

### SciPy and other runtime dependencies

Add SciPy to `[project.dependencies]` because FaST-LMM imports it directly.
Determine and test an honest lower bound. If Python 3.14 needs a different lower
bound, express it with a Python-version marker without unnecessarily raising
the bound on Python 3.10 through 3.13.

The historical `jun25` branch is evidence that SciPy and statsmodels
compatibility needs explicit testing, but it is not an implementation base for
this work. Do not cherry-pick its final dependency metadata: that branch limits
Python to `<3.13`, points statsmodels at an unreleased Git commit, and constrains
the optional BGEN stack to older releases. Reproduce the compatibility problem
on the `py314` branch, identify which supported Python/dependency combinations
are affected, and solve it using stable published releases and truthful bounds.
Record the tested SciPy/statsmodels combinations in the implementation change.

Adding SciPy as a runtime dependency is required, but retaining SciPy as a
build-system requirement is not. Remove it from `[build-system].requires` when
migrating to `uv_build`; restore a build dependency only if an isolated build
demonstrates that it is genuinely needed and document why.

Verify that stable releases of every direct and transitive dependency resolve,
install, and pass tests on every supported Python version. In particular,
check:

- NumPy
- SciPy
- pandas
- Matplotlib
- scikit-learn
- cloudpickle
- statsmodels
- psutil
- `pysnptools`
- `fastlmmclib`
- optional BGEN dependencies (`cbgen` and `bgen-reader`)
- development, test, build, and documentation dependencies

Where Python 3.14 requires a newer dependency, use a Python-version marker so
older Python versions retain their existing lower bound when possible. Prefer
stable releases. Prereleases may be used only as a documented temporary bridge
during development and must not be needed for release acceptance.

Before release, explicitly resolve the optional BGEN status on Python 3.14.
If stable compatible `cbgen` and `bgen-reader` artifacts are available, test
and support the `bgen` extra normally. If they are not available, do not leave
the outcome implicit: exclude or mark the extra as unsupported on Python 3.14
with accurate dependency markers, documentation, release notes, and CI that
tests both the supported extra combinations and the expected exclusion. Do not
use an unpublished checkout or prerelease wheel to claim final support.

### Migrate the build backend to `uv_build`

Replace setuptools with `uv_build` as part of this release. FaST-LMM is a
pure-Python top-level package and no active root-package Cython or extension
build has been identified, so it fits `uv_build`'s supported project type.
`bed-reader` uses maturin because it contains a Rust extension; that does not
apply to FaST-LMM.

At implementation time, recheck the current compatible `uv_build` minor line,
pin a lower and upper bound as recommended by Astral, and align it with the
reviewed `uv` tool version. As of this spec's toolchain review, the documented
configuration is:

```toml
[build-system]
requires = ["uv_build>=0.11.32,<0.12"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "fastlmm"
module-root = ""
```

The migration must:

1. Remove the setuptools build backend and all `[tool.setuptools]`
   configuration.
2. Remove NumPy, SciPy, Cython, wheel, and setuptools from build requirements.
   Restore a build requirement only if a verified active build step needs it
   and document the reason.
3. Configure the existing flat-layout `fastlmm` module with
   `module-name = "fastlmm"` and `module-root = ""`, plus any required source
   inclusion or exclusion rules under `[tool.uv.build-backend]`.
4. Translate the existing package-data behavior for licenses, authorship
   files, RST files, sample datasets, the hashdown JSON file, and the bundled
   platform executables and manual.
5. Exclude top-level tests, generated output, documentation builds, notebooks,
   nested obsolete packaging files, caches, and other repository-only material
   unless intentionally required in the sdist. Review package-internal test
   modules and data individually; retain them only where runtime or documented
   compatibility requires them.
6. Build both wheel and sdist with `uv build --no-sources` so unpublished local
   source overrides cannot make the release build pass.
7. Build a wheel from the generated sdist and compare it with the wheel built
   directly from the checkout. Both paths must contain the same intended
   runtime files and metadata.
8. Compare the new wheel and sdist file manifests with artifacts produced by
   the last setuptools release. Investigate every added or removed file and
   record intentional differences in the implementation pull request.
9. Run the isolated artifact tests against both outputs on Python 3.10 and
   3.14.

Do not retain a parallel setuptools build path or compatibility fallback after
the migration passes these checks. `uv_build` is the canonical backend for the
release.

## Development Dependency Groups

Replace `[tool.uv].dev-dependencies` with standardized PEP 735
`[dependency-groups]`. Separate test, lint, and documentation tools and compose
them into a default development group, for example:

```toml
[dependency-groups]
test = [
    "pytest",
    "pytest-cov",
    "pytest-datadir",
    "pytest-doctestplus",
]
lint = ["ruff"]
docs = [
    "sphinx",
    "sphinx-rtd-theme",
]
dev = [
    { include-group = "test" },
    { include-group = "lint" },
    { include-group = "docs" },
]
```

Reconcile this with the current `[project.optional-dependencies].dev` extra:

- Development-only tools should live in dependency groups and should not be
  published as end-user package requirements.
- Retain a published `dev` extra only if there is evidence that downstream
  users rely on `fastlmm[dev]`; otherwise remove it and document the new local
  setup command.
- Keep feature extras such as `bgen` in `[project.optional-dependencies]`.

## Package Metadata

Update `pyproject.toml` to:

- Add `Programming Language :: Python :: 3.14`.
- Keep the classifiers for Python 3.10 through 3.13.
- Keep `requires-python = ">=3.10"` unless `bed-reader` changes its supported
  range before implementation. If it does, explicitly reconcile the projects
  and this spec rather than silently diverging.
- Declare the marked NumPy runtime requirements and direct SciPy requirement.
- Raise the `pysnptools` and `fastlmmclib` minimum versions to their first
  Python 3.14-compatible releases.
- Replace deprecated license-table metadata with PEP 639 metadata:

  ```toml
  license = "Apache-2.0"
  license-files = ["LICENSE.md"]
  ```

- Change project URLs from HTTP to HTTPS where HTTPS is available.
- Use clear conventional URL labels such as `Homepage`, `Documentation`,
  `Issues`, and `Source`.

Build and inspect both artifacts to verify that metadata and included files are
correct, including the license, authorship files, documentation, executables,
and sample data.

## Code Compatibility

Run the full suite under Python 3.14 and fix incompatibilities in production
code, tests, doctests, examples, and packaging. Pay particular attention to:

- APIs removed or deprecated in Python 3.14.
- NumPy 2.x behavior and removed aliases.
- Changed behavior in current SciPy, pandas, Matplotlib, scikit-learn, and
  statsmodels releases.
- Native-extension loading and platform-specific behavior from
  `fastlmmclib`, `pysnptools`, and optional BGEN dependencies.
- Floating-point or textual expected-output differences. Update expectations
  only after confirming that the new result is correct and does not conceal a
  numerical regression.
- The legacy `fastlmm/pyplink/setup.py`, which imports `distutils`. Determine
  whether it is still shipped or invoked, then remove, migrate, or exclude it
  as appropriate. Do not add a compatibility shim solely for Python 3.14.

Compatibility fixes must preserve behavior on Python 3.10 through 3.13.

The `Legend.legendHandles` failure reported in
[FaST-LMM issue #48](https://github.com/fastlmm/FaST-LMM/issues/48) has already
been changed in the current source to the modern `legend_handles` spelling.
Add a headless plotting regression test that reaches this legend-handling path
under the supported Matplotlib versions. Close the issue only after that test
passes in CI; no additional compatibility workaround should be added unless
the supported version matrix demonstrates that one is required.

## Test-Suite Modernization

Establish one documented command as the complete local and CI test entry point.
Before choosing it:

1. Inventory the suites run by `tests/test.py`.
2. Compare them with tests collected by `pytest` from the repository root.
3. Confirm whether the doctests configured in `pyproject.toml` are currently
   executed.
4. Add missing legacy suites to normal pytest discovery where practical.
5. If an immediate migration would omit coverage, run both the legacy entry
   point and pytest temporarily and document why.

The final required CI must demonstrably run unit tests, integration tests,
configured doctests, and relevant examples. Retire the bespoke aggregator only
after pytest provides equivalent coverage.

Update Ruff from 0.7.1 to a reviewed current version and use that exact version
locally and in CI. Run the required Ruff check once in its own job. Remove the
always-ignored "latest Ruff" step; use dependency-update pull requests to
advance the reviewed pin instead.

## Continuous Integration

### Runner and Python matrix

Replace deprecated macOS runner labels with explicit current Intel and Apple
Silicon coverage. Following current `bed-reader`, the required test matrix is:

| Operating system and architecture | Runner | Python versions |
| --- | --- | --- |
| Linux x86-64 | `ubuntu-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| Windows x86-64 | `windows-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| macOS Apple Silicon | `macos-15` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| macOS Intel | `macos-15-intel` | 3.10, 3.11, 3.12, 3.13, 3.14 |

Recheck runner availability immediately before implementation. Preserve both
macOS architectures even if GitHub changes the exact labels.

Do not allow Python 3.14 jobs to fail without failing CI. Retain
`fail-fast: false` so cross-platform failures remain visible.

### Job structure

Split CI into focused jobs:

1. **Lint:** run the pinned Ruff version once on Ubuntu.
2. **Test matrix:** run the complete tests on Python 3.10 through 3.14 across
   all four runner types.
3. **Minimum dependencies:** test `lowest-direct` resolution on the applicable
   boundary interpreters.
4. **Build:** run `uv build --no-sources` once and produce both wheel and
   sdist.
5. **Artifact tests:** install and test the exact wheel and sdist in isolated
   Python 3.10 and 3.14 environments.
6. **Dependency freshness:** on the monthly schedule, resolve the newest stable
   dependencies and run tests without changing the committed lockfile.

Do not run Ruff and package builds redundantly in every matrix entry. Upload
both wheel and sdist under an artifact name that describes their contents.

For every matrix test entry, CI must:

1. Select the requested interpreter with `setup-uv`.
2. Synchronize the frozen project environment with all feature extras and
   required dependency groups.
3. Run the canonical full test entry point with `uv run --frozen`.

### Artifact verification

Create a small, committed smoke-test script and test both built artifacts
without the repository on `PYTHONPATH`, following this pattern:

```bash
uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
```

The smoke test must at minimum verify:

```python
import fastlmm
import fastlmm.association
import fastlmm.inference
```

Where practical, it should also load a tiny packaged dataset and perform one
inexpensive calculation so package-data and native dependency problems are
detected. The Python 3.14 artifact tests must consume published releases of
`pysnptools` and `fastlmmclib` for final acceptance.

### Workflow reliability and security

- Set top-level or per-job `permissions: contents: read`; grant nothing more to
  ordinary CI.
- Pin every action, including GitHub-authored actions, to a reviewed full
  commit SHA with a release comment.
- Enable Dependabot updates for GitHub Actions so reviewed pins remain current.
- Upgrade `actions/checkout`, `actions/upload-artifact`, and other actions to
  supported current releases before pinning them.
- Add workflow concurrency that cancels superseded runs for the same pull
  request or branch.
- Add reasonable `timeout-minutes` values so hung numerical tests do not consume
  runners indefinitely.
- Avoid duplicate CI for the same commit by selecting deliberate `push` branch
  filters while retaining `pull_request`, monthly `schedule`, and
  `workflow_dispatch` triggers.
- If cache pruning materially reduces storage or transfer, run
  `uv cache prune --ci` at the end of jobs that populate the cache.

## Release Workflow

Add a separate release workflow using PyPI Trusted Publishing rather than a
long-lived API token.

The workflow must:

1. Trigger only from a version tag, or from a published GitHub release backed
   by that version tag. For FaST-LMM, PySnpTools, and fastlmmclib, every PyPI
   version must have a matching immutable Git tag pointing to the exact source
   commit used for the artifacts, addressing
   [fastlmmclib issue #2](https://github.com/fastlmm/fastlmmclib/issues/2).
2. Build wheel and sdist once in a job with only read permissions.
3. Run the isolated artifact tests against those exact files.
4. Upload those exact tested files as GitHub workflow artifacts.
5. Publish the downloaded files from a separate job rather than rebuilding.
6. Use a protected `pypi` GitHub environment with manual approval.
7. Grant `id-token: write` only to the publishing job.
8. Use PyPI Trusted Publishing/OIDC and retain the generated attestations.
9. Never publish from pull-request workflows or with an untrusted checkout.

Configure the matching Trusted Publisher in PyPI and revoke obsolete stored
PyPI tokens after the new workflow is proven.

## Local Verification

Before opening or merging the implementation change:

1. Install the pinned `uv` version.
2. Confirm `uv lock --check` succeeds and the committed lockfile is unchanged.
3. Run the pinned Ruff check.
4. Create clean environments for Python 3.10 and Python 3.14.
5. In each environment, synchronize all feature extras and required dependency
   groups and run the canonical complete test command.
6. Run the lower-bound tests on the applicable Python boundary versions.
7. Run `uv build --no-sources`.
8. Build a wheel from the generated sdist and compare its manifest and
   metadata with the wheel built directly from the checkout.
9. Compare the new artifact manifests with the last setuptools-built release
   and account for every difference.
10. Inspect wheel and sdist metadata for Python requirements, classifiers,
   license metadata, NumPy and SciPy markers, and prerequisite lower bounds.
11. Inspect wheel and sdist contents for all intended package data.
12. Install and exercise each artifact in isolated Python 3.10 and 3.14
    environments.
13. Let the complete CI matrix pass on all four operating-system/architecture
    combinations.
14. Verify the tutorial-data retrieval smoke test in PySnpTools, the
    Matplotlib legend regression test in FaST-LMM, and the documented BGEN
    support or exclusion behavior on Python 3.14.
15. Verify that each release artifact version and source commit correspond to
    the release's version tag.

Python 3.10 and 3.14 are the required local boundary checks; CI remains the
authority for intermediate Python versions and hosted runner platforms.

## Documentation and Release Work

- Add a changelog or release-note entry announcing Python 3.14 support and the
  retained Python 3.10-through-3.14 range.
- Mention the new minimum versions of `pysnptools` and `fastlmmclib` so users
  know to update pinned environments.
- Mention that Python 3.14 resolves NumPy 2.3.5 or newer.
- Document `uv` as the standard contributor setup and command interface.
- Document the canonical lint, test, lower-bound test, build, and artifact-test
  commands.
- Add a "Policy on AI-assisted development and contributions" section to the
  root `README.md` files of both FaST-LMM and PySnpTools. Model it on the policy
  used by the maintainer's Rust projects: permit AI tools as productivity aids
  for drafting, exploration, and refactoring; require every contributed code
  and documentation change to be reviewed, edited, and validated by a human;
  and state that AI does not replace design judgment, testing, or human
  responsibility for correctness.
- In each policy section, link to that repository's root `AGENTS.md` as the
  published instructions and constraints supplied to AI tools during
  development. Keep `AGENTS.md` as the substantive source of agent guidance;
  do not duplicate its detailed contents in the README.
- Update installation or contributor documentation wherever supported versions
  or obsolete pip/virtualenv commands are stated.
- Choose the FaST-LMM release version according to the project's normal release
  policy; this spec does not prescribe a version number.
- Publish only after the prerequisite `pysnptools` and `fastlmmclib` releases
  are available and final clean-install checks pass against them.

## Acceptance Criteria

Python 3.14 support and the associated toolchain update are complete when all
of the following are true:

- Package metadata advertises Python 3.10 through 3.14.
- Python 3.14 resolves NumPy 2.3.5 or newer; Python 3.10 through 3.13 retain the
  existing NumPy lower bound.
- NumPy and SciPy are declared as direct runtime dependencies with tested lower
  bounds, and unnecessary build requirements have been removed.
- `uv_build` is the sole build backend; setuptools configuration and fallback
  paths are removed.
- Direct-checkout and sdist-derived wheels have equivalent intended contents,
  and every artifact-manifest difference from the last setuptools release has
  been reviewed and explained.
- Published Python 3.14-compatible releases of `pysnptools` and `fastlmmclib`
  exist, and FaST-LMM requires at least those versions.
- PySnpTools tutorial data referenced by its hashdown manifest is available,
  integrity-checked, and covered by a retrieval smoke test.
- fastlmmclib's native extension is regenerated with a current compatible
  Cython and its distributed wheels pass clean Python 3.14 artifact tests.
- All required dependencies install from stable releases on Python 3.14. The
  optional BGEN stack either does so and passes its tests, or is accurately
  excluded from Python 3.14 with the limitation documented and tested.
- A universal `uv.lock` is committed, ordinary CI uses frozen resolution, and
  separate required lower-bound and scheduled freshness tests pass.
- CI uses a reviewed pinned `uv`, `uv run`, caching, and no manual environment
  activation.
- The full CI matrix passes for Python 3.10 through 3.14 on Linux, Windows,
  Intel macOS, and Apple Silicon macOS.
- Ruff, the complete unit and integration suite, configured doctests, and
  package builds pass.
- A headless regression test covers the corrected Matplotlib
  `legend_handles` path reported in FaST-LMM issue #48.
- The exact wheel and sdist install and pass smoke tests in isolated Python
  3.10 and 3.14 environments.
- No Python 3.14 job is experimental or allowed to fail.
- Current PEP 639 license metadata and HTTPS project URLs are present.
- GitHub Actions use minimal permissions and reviewed full-SHA pins.
- A protected Trusted Publishing workflow publishes the already-tested
  artifacts without a long-lived PyPI token.
- Every published fastlmmclib, PySnpTools, and FaST-LMM version in this release
  sequence has a matching version tag at its exact build source commit.
- The root READMEs for FaST-LMM and PySnpTools contain the required
  human-accountability policy for AI-assisted development and link to their
  respective root `AGENTS.md` files.
- Release notes and contributor documentation describe the support and
  toolchain changes.

## Primary References

- [Installing and managing Python with `uv`](https://docs.astral.sh/uv/guides/install-python/)
- [Using `uv` in GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/)
- [`setup-uv` action documentation](https://github.com/astral-sh/setup-uv)
- [`uv` dependency and dependency-group management](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [`uv` resolution and lower-bound testing](https://docs.astral.sh/uv/concepts/resolution/)
- [`uv_build` configuration and file inclusion](https://docs.astral.sh/uv/configuration/build-backend/)
- [Building distributions with `uv`](https://docs.astral.sh/uv/concepts/projects/build/)
- [PEP 735 dependency groups](https://packaging.python.org/en/latest/specifications/dependency-groups/)
- [Current `pyproject.toml` and PEP 639 guidance](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [PyPI Trusted Publishing from GitHub Actions](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [GitHub-hosted runner images](https://github.com/actions/runner-images)
- [GitHub Actions security guidance](https://docs.github.com/en/code-security/tutorials/secure-your-organization/protect-against-threats)

## GitHub Issue and Pull-Request Triage

The open issue and pull-request review for this release produced the explicit
requirements above. Keep the remaining work out of the Python 3.14 critical
path unless implementation reveals a direct compatibility failure:

- [FaST-LMM issue #55](https://github.com/fastlmm/FaST-LMM/issues/55), a
  multi-phenotype usage and output-size support question.
- [FaST-LMM issue #52](https://github.com/fastlmm/FaST-LMM/issues/52), a
  request to document how the synthetic test dataset was generated.
- [FaST-LMM issue #51](https://github.com/fastlmm/FaST-LMM/issues/51), a
  performance report for analyses exceeding five million SNPs.
- [FaST-LMM issue #26](https://github.com/fastlmm/FaST-LMM/issues/26), a
  request to batch large numbers of phenotypes to reduce memory use.

Do not merge either currently open FaST-LMM pull request into this release:

- [Pull request #15](https://github.com/fastlmm/FaST-LMM/pull/15) is a stale,
  conflicting Manhattan-plot enhancement without current validation.
- [Pull request #7](https://github.com/fastlmm/FaST-LMM/pull/7) is a stale,
  unrelated Docker contribution containing substantial notebook, checkpoint,
  generated, and binary data.

PySnpTools and fastlmmclib had no open pull requests at the time of this
review. Recheck all three repositories immediately before release rather than
assuming this snapshot remains current.

### Post-release closure checklist

After the Python 3.14 release sequence is complete and its published artifacts
have passed final acceptance, review and close these items with a comment
linking to the relevant release, test, tag, or superseding decision:

- [PySnpTools issue #10](https://github.com/fastlmm/PySnpTools/issues/10):
  close after the released package retrieves all five synthetic tutorial files
  from the immutable `bed-sample-files` revision, verifies their hashes, and
  successfully opens the BED/BIM/FAM dataset.
- [FaST-LMM issue #48](https://github.com/fastlmm/FaST-LMM/issues/48): close
  after the corrected Matplotlib `legend_handles` path is covered by a
  passing headless regression test in the released version.
- [fastlmmclib issue #2](https://github.com/fastlmm/fastlmmclib/issues/2):
  close after the Python 3.14-compatible fastlmmclib PyPI release has a
  matching immutable Git tag at the exact artifact source commit.
- [FaST-LMM pull request #15](https://github.com/fastlmm/FaST-LMM/pull/15):
  close without merging. Explain that the old conflicting plotting enhancement
  was not part of the compatibility release and invite a new focused pull
  request against current code if the feature is still wanted.
- [FaST-LMM pull request #7](https://github.com/fastlmm/FaST-LMM/pull/7):
  close without merging. Explain that the stale Docker change and its bundled
  notebook, checkpoint, generated, and binary files do not fit the current
  release or repository packaging approach.

Do not close FaST-LMM issues #55, #52, #51, or #26 merely because the Python
3.14 release completed. They describe support, documentation, performance, or
memory-scaling work that this release does not resolve.

## Out of Scope

- Dropping any Python version from 3.10 through 3.13 while `bed-reader`
  continues to support it.
- Adding support for Python versions newer than 3.14.
- Replacing `pysnptools`, `fastlmmclib`, or the optional BGEN stack.
- Rewriting native dependencies in Rust merely because `uv` is written in
  Rust.
- Unrelated modernization or refactoring discovered during compatibility work.
- The deferred FaST-LMM issues listed in the GitHub triage section, unless a
  release-gating Python 3.14 regression is demonstrated.
