# Releasing FaST-LMM

This is the standing release process for FaST-LMM. Short-term release plans and
compatibility investigations belong in `specs/`; keep this file limited to the
process that should apply to every release.

The maintainer chooses the version and explicitly approves publication. A
release must be built from a clean commit on the repository's default branch,
and its `vX.Y.Z` tag must point to that exact commit.

## Release order

Publish and verify any required PySnpTools and `fastlmmclib` releases before
final FaST-LMM qualification. Update FaST-LMM's minimum dependency versions to
the first compatible published releases.

Final qualification must install those prerequisites from PyPI, not from
sibling source checkouts, Git dependencies, or prereleases.

## One-time repository setup

- Configure a PyPI Trusted Publisher for the `fastlmm/FaST-LMM` repository,
  `.github/workflows/release.yml`, and a protected GitHub `pypi` environment.
- Require maintainer approval for the `pypi` environment and restrict it to
  version tags.
- Give the publish job only the permissions it needs, including
  `id-token: write`. Other jobs need only `contents: read`.
- Pin third-party GitHub Actions to reviewed full commit SHAs and pin `uv` to a
  reviewed version.
- Pass the distributions built and tested by CI to the publish job as an
  artifact. Never rebuild them in the publish job.

The release workflow must publish with PyPI Trusted Publishing and retain the
generated attestations. Do not store PyPI passwords or API tokens in the
repository or GitHub Actions.

## Prepare the release

1. Start from a clean release branch based on the current default branch.
2. Review open issues, pull requests, and dependency advisories for anything
   that affects the release.
3. Set the version in `pyproject.toml` and update the release notes with the
   release date, compatibility changes, deprecations, and user-visible fixes.
4. Confirm that the required published PySnpTools and `fastlmmclib` versions
   install on every supported Python version and platform.
5. Confirm that dependency bounds and Python-version markers describe versions
   actually tested in CI. Exercise the declared direct minimums on the oldest
   and newest supported Python versions:

   ```console
   uv venv --python 3.10 .minimum-310
   uv pip install --python .minimum-310/bin/python --resolution lowest-direct --editable ".[bgen]"
   cd tests
   ../.minimum-310/bin/python test.py
   cd ..
   uv venv --python 3.14 .minimum-314
   uv pip install --python .minimum-314/bin/python --resolution lowest-direct --editable ".[bgen]"
   cd tests
   ../.minimum-314/bin/python test.py
   cd ..
   ```
6. Regenerate and commit `uv.lock`, then verify it:

   ```console
   uv lock --check
   ```

7. Run the complete canonical test suite and all supported optional-dependency
   suites from the locked environment:

   ```console
   uv sync --frozen --all-extras
   cd tests
   uv run --frozen --no-sync python test.py
   cd ..
   ```

8. Run representative end-to-end association and inference tests. Investigate
   every change in numerical results, tolerances, ordering, dtypes, or result
   schemas before updating expected output.
9. Execute the maintained notebooks from start to finish in clean environments
   and inspect their results. Build the documentation from its sources and
   check links and examples. Create the locked Python 3.14 notebook environment
   with:

   ```console
   UV_PROJECT_ENVIRONMENT=.venv-notebook314 uv sync --python 3.14 --frozen --all-extras --group notebook
   ```

   The maintained public notebooks are the four linked from `README.md`.
   Record any deliberately skipped machine-specific or multi-hour example and
   why it was not part of the routine execution pass.
10. Build the source distribution and wheel without local source overrides:

    ```console
    uv build --no-sources
    ```

11. Inspect both artifact manifests for required metadata, licenses, sample
    data, hashdown files, native executables, and package data. Confirm that
    development files, caches, notebooks, and generated output are absent unless
    intentionally distributed.
12. Install and test the exact wheel and source distribution in clean
    environments on the oldest and newest supported Python versions, outside
    the source checkout and without the repository on `PYTHONPATH`.
13. Verify the installed `fastlmmclib` native artifact on every supported
    operating-system and architecture combination.
14. Wait for all required CI jobs to pass on every supported operating system
    and Python version. Resolve warnings that indicate a compatibility,
    packaging, numerical, or security problem.

If the repository does not yet have the locked `uv` workflow assumed above,
finish that migration before releasing rather than substituting an unreviewed
release path.

## Publish

1. Merge the reviewed release change into the default branch and confirm that
   it is clean and up to date.
2. Confirm that the version is not already present on PyPI and that the tag does
   not already exist.
3. Create and push an annotated tag at the release commit:

   ```console
   git tag -a vX.Y.Z -m "FaST-LMM X.Y.Z"
   git push origin vX.Y.Z
   ```

4. Verify that the tag workflow builds and tests the exact artifacts intended
   for publication.
5. Review the workflow summary and approve its protected `pypi` environment.
6. Confirm that PyPI received both the wheel and source distribution and that
   their attestations are present.
7. Create the matching GitHub release from the release notes.

## Verify the published release

- Install `fastlmm==X.Y.Z` and its dependencies from PyPI into a fresh
  environment with no sibling checkout on `PYTHONPATH`.
- Run representative association and inference smoke tests against the
  published artifacts on the oldest and newest supported Python versions.
- Run the relevant InstallTest scenarios.
- Build or deploy the website from its source repository and verify the rendered
  documentation, examples, downloads, and tutorial links.
- Close or update the issues and pull requests resolved by the release, linking
  to the published version or its CI evidence.

## Failed releases

Do not move or replace a published version tag, and do not upload different
files under an existing version. If a published release is unusable, yank it on
PyPI with a concise reason and publish a corrected version. Preserve the failed
release's tag and evidence for traceability.

TestPyPI may be used through a separate Trusted Publisher when a publishing
workflow itself needs qualification. It does not replace clean local artifact
tests or final testing against the real PyPI dependency ecosystem.
