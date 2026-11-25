# Annual FaST-LMM ecosystem update

This coordinates the repositories; each package's `RELEASING.md` remains the
authority for its release gates. Record exact commits, CI runs, artifacts,
and published versions for each annual update.

## Release order

1. Publish and verify bed-reader, including the new interpreter's wheels.
2. Publish and verify fastlmmrslib, including installed numerical tests.
3. Publish and verify PySnpTools against published bed-reader artifacts.
4. Update FaST-LMM's bounds and lockfile against the published prerequisites,
   then qualify and publish FaST-LMM when its behavior or requirements changed.
5. Update InstallTest's exact published versions and run its platform matrix.

Release fetch-data first when bed-reader's Rust tests need its new behavior.
cloud-file has an independent release cycle. Do not release a package merely
because a development lockfile changed.

## Preparation and qualification

- Read each repository's `AGENTS.md`, `RELEASING.md`, workflow matrices, and
  changelog. Preserve each file's line endings and existing TODO comments.
- Inspect direct dependency lower bounds and changed upstream defaults. Use
  Python-version markers rather than raising requirements on older interpreters
  unnecessarily. Keep optional formats separate from core support.
- Use the uv version pinned in CI, external environments under `~/.cache`, and
  deliberate lockfile updates. Do not qualify against sibling checkouts.
- Run the canonical suites, doctests, optional-format suites, strict Sphinx
  build, link checks, and sample-manifest integrity checks required by the
  package. Document every existing skip and unavailable environment.
- Build wheel and source distribution in isolation. Compare direct and
  source-derived wheels where reproducible, inspect metadata/licenses/package
  data, and install both artifacts outside the checkout on boundary Python
  versions. Use published binary dependencies.
- For Rust, test the declared MSRV and stable, retain CI's exact Clippy flags,
  and keep dependency resolution compatible with the MSRV.

## CI and branch protection

The package repositories run full CI for pull requests, on demand, and monthly.
They do not repeat full CI on the merge push. A final `CI result` job uses
`if: always()` and requires every dependency job to succeed. Documentation
deployment and tag-triggered artifact qualification have separate workflows.

Before merging, inspect protection on `main`: require `CI result`, require the
branch to be current with `main`, and prohibit force pushes. As audited on
2026-11-14, these settings cover bed-reader, fastlmmrslib, PySnpTools,
FaST-LMM, cloud-file, and fetch-data. Admin enforcement is disabled; use the
normal checked PR path anyway. InstallTest currently has no branch protection
and tests pushes plus its weekly schedule.

Monitor every active PR and workflow run; report failures and completion
without waiting for the maintainer to discover them. Before merging, verify
the checks apply to the exact head commit and merge with an expected-head SHA.
Do not treat a completed run with failed or skipped required jobs as green.

## Notebook checks

- PySnpTools: execute `tutorial`, `paper`, and any affected optional-format
  notebooks in a clean environment; inspect numerical output changes.
- FaST-LMM: follow `tests/notebooks.toml`, including the full active notebooks,
  the explicitly qualified SingleSnpScale prefix, and documented historical
  exclusions. Saved outputs do not replace fresh execution.
- The helpers in `tools/annual-update/` execute copies and compare outputs.
  Keep environments and results outside the repository. Review changes before
  using the explicit output refresher; preserve unexecuted suffix cells and
  file line endings. Never refresh expected results merely to hide a failure.

## Attribution and instructions

Before committing, inspect commit messages, PR bodies, and release notes for
AI attribution. Do not add AI coauthor trailers or generated-by credits.
Keep that policy explicit in every repository's `AGENTS.md`.

Current Codex and Claude Code can share repository `AGENTS.md` directly.
Claude Code's native support requires v2.1.277 or later; older clients need
an import from their personal instructions. Keep repository guidance portable
without personal home files. On both desktop and laptop, both WSL2 and Windows
use matching `~/.codex/AGENTS.md` files and personal `~/.claude/CLAUDE.md`
imports. Do not commit one-line repository CLAUDE.md stubs.

## Publication gate

Only publish after the reviewed release branch is merged, all required CI
passes, the version is absent from the index, and the working tree is clean.
The maintainer pushes the release tag and approves the protected `pypi`
environment. Agents prepare artifacts and commands unless explicitly asked to
publish. Never replace a published tag or upload different files under an
existing version.

For PySnpTools 1.0.1, after merging its release PR and updating local `main`:

```bash
git tag -a v1.0.1 -m "PySnpTools 1.0.1"
git push origin v1.0.1
```

The tag workflow builds and tests distributions, then waits for the maintainer's
PyPI approval. Verify both published artifacts before continuing:

```bash
# In FaST-LMM, using its pinned uv version:
uv lock --upgrade-package pysnptools
uv lock --check
```

Confirm the lockfile selects 1.0.1 without unrelated upgrades, rerun the
relevant FaST-LMM validation, and change InstallTest's `PYSNPTOOLS_VERSION`
to `1.0.1`. No FaST-LMM release is needed for this lockfile-only follow-up.

## Housekeeping and next update

- Review the branch inventory with the maintainer. Reachability does not prove
  that a squashed branch is still needed, and age does not prove safe deletion.
  Preserve backup branches for at least a week after history acceptance; delete
  only branches the maintainer chooses.
- Keep monthly grouped Dependabot coverage for GitHub Actions and each repo's
  Cargo/uv manifests. Review Rust installer refs manually to retain the MSRV.
- For November 2027: investigate Python 3.16 and decide whether to retain 3.11;
  verify actual Intel macOS runner availability and plan cross-build/Rosetta
  qualification before changing support; assess fetch-data's ureq 3 migration
  and replacement of anyinput/proc-macro-error; decide FaST-LMM issue #26's
  multiple-phenotype batching scope. These are future investigations, not
  changes authorized for the current compatibility release.
