# Annual update scope and progress

Approved November 13, 2026: A through I; J (phenotype batching) excluded.
E was explained and subsequently approved. The maintainer will post the gx2
issue; agents must not post it. Publication of package releases remains a
separate maintainer action or explicit authorization.

Target Python 3.11.5 through 3.15 for FaST-LMM/PySnpTools (bed-reader and
fastlmmclib can retain a 3.11 floor). Proposed versions: bed-reader 1.2.0,
fastlmmclib 1.0.0, PySnpTools 1.0.0, FaST-LMM 1.0.0. The maintainer subsequently
selected fastlmmrslib 0.1.0 as the Rust successor's first release candidate.

## Work items

- A: Complete Python 3.15 support, dependency refresh, optional HDF5/BGEN,
  and wheel/sdist acceptance across supported platforms. Extensive Linux
  validation is already recorded in the session's `/tmp` logs. Published
  prerequisite artifacts and final cross-platform acceptance remain required.
- B: bed-reader cloud-file 0.3.1 migration implemented on `py15`.
- C: NaN integer-property conversion fixed without mutating input. Four
  regressions failed before the fix and pass afterward, including read-only
  arrays. Python 3.11 installed-wheel regression tests and Python 3.15 source
  regressions/local/cloud coverage pass. Full checks and rebuilt artifacts
  remain required.
- D: Replace PySnpTools' six queue import with the standard library; remove
  FaST-LMM's Python 2 URL-import fallback and direct six declarations.
  Source edits are prepared, and the multithreaded map-reduce test passes.
  Refresh locks, test clean installations, and coordinate the published
  PySnpTools minimum before qualifying FaST-LMM without its workaround.
- E: Integrate PR #64's five reviewed action versions across coordinated
  projects. Pins are updated locally; workflow execution remains required.
- F: Move FaST-LMM and PySnpTools documentation to official Pages artifacts.
  Both currently deploy `/docs` from the default branch without custom domains.
  Preserve URLs; verify deployment before removing tracked HTML and stale-doc
  comparison gates. Keep strict documentation builds in CI and release checks.
- G: GitHub default branch renamed from master to main on November 13.
  Local tracking, PR #64's base, and Pages' source all verified as main.
  No branch protection or repository rulesets were configured. Source links
  and local-clone migration instructions are prepared; commit and verify
  documentation changes as part of the maintenance work.
- H: Add Windows ARM64 native wheels and tests, followed by downstream
  artifact installs. This is distinct from the existing Linux ARM64 coverage.
- I: Try fastlmmrslib as a Rust replacement for fastlmmclib's full quadratic-form API.
  Published gx2 1.5 failed independent correctness checks; see
  `QUADRATIC_FORM_EVALUATION.md`. Preserve diagnostics and provenance, use
  per-call state, and qualify independently before selecting a production
  replacement. Bundled fastlmmc executable retirement is separate and is not
  automatically authorized by this work item.

## Current CI findings

- PySnpTools annual changes are committed on `annual-update-2026` and pushed
  as draft PR #18. CI run 43516916408 tests Python 3.11–3.15; Python 3.15
  qualifies core separately from optional HDF5/BGEN backends. The superseded
  documentation-only PR #17 is closed, with its changes included in #18.
  Branch pushes now trigger CI only on main, avoiding duplicate push/PR suites.
- Obsolete baseline runs were consuming four of the five occupied macOS slots.
  Cancellation was requested for PySnpTools runs 43510093198, 43510079801,
  and 43509918159, and bed-reader runs 43510271894 and 43509926644.
  Current annual qualification runs remain intact.
- bed-reader run 43503954560 includes the local zero-byte-file regression fix
  and native Windows ARM64 coverage. No test was skipped for the earlier
  remote-download HTTP 501 failure.
- fastlmmclib run 43503649643 passed Windows ARM64 Python 3.11–3.13 but its
  Python 3.14 interpreter download timed out. Retry that failed job after the
  active run finishes; do not hide the failure or remove the architecture.
- fastlmmrslib run 43511476395 passed all 38 native jobs. The versioned 0.1.0
  candidate passes minimum-Rust tests/strict Clippy, Rust publication dry-run,
  strict metadata validation, and installed direct/sdist-derived wheel tests
  on Python 3.11 and 3.15. Final native qualification remains required.
- FaST-LMM Pages PR #65 is merged and its documentation deployment passed.
  Live index, search, general index, stylesheet, and source links returned HTTP
  200. Remove tracked HTML and freshness comparisons; retain strict builds.
- bed-reader's source-code head passed all 58 native jobs. Release preparation
  is in PR #36; the final preparation head must pass qualification before tagging.
- PySnpTools PR #18 completed with 29 passes and nine failures, all confirmed
  missing published bed-reader wheels for Python 3.15 or Windows ARM64.
- A fresh FaST-LMM frozen notebook environment passes Ruff, but strict Sphinx
  fails importing the currently published PySnpTools 0.5.15: its map-reduce
  runner still imports six. The prepared PySnpTools 1.0.0 removes that import.
  Do not reintroduce six or weaken autodoc; publish the prerequisite and then
  raise the minimum, regenerate the lock, and rerun strict documentation.

## Release sequence

Validate and publish bed-reader and fastlmmrslib 0.1.0 first;
validate and publish PySnpTools against those published dependencies; update
FaST-LMM minimums and qualify its final artifacts against published packages.
Local combined-artifact experiments are provisional evidence only.

## Latest qualification

- All eight workspace GitHub default branches now use `main`; local default
  branches and Pages sources were migrated and branch-trigger fixes prepared.
- PySnpTools' installed core wheel passes 444 canonical tests with 11 expected
  HDF5 skips and no six package installed. External checkout data fixtures were
  staged because the wheel intentionally excludes examples and datasets.
- The Rust prototype matches probabilities and fault codes across 309 captured
  and synthetic cases. Early-failure and exact-support diagnostic traces can
  differ. Direct and sdist-derived Python 3.11/3.15 wheels pass eight independent
  probability/binding tests. Six Rust tests and minimum-Rust strict Clippy pass.
- FaST-LMM's complete canonical suite passes with the installed Rust prototype
  substituted only in the test process: 181 tests, five expected HDF5 skips.
  Source is unchanged; production dependency selection awaits published 0.1.0
  artifacts and final acceptance against those artifacts.
- Original third-party notices in fastlmmrslib are preserved verbatim, including
  archival whitespace. No license wording was silently rewritten for linting.
