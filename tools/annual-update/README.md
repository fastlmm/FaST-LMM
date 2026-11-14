# Annual update helpers

Reusable helpers retained from the November 2026 update. The coordinated
process is in [the annual update runbook](../../specs/ANNUAL_UPDATE_RUNBOOK.md).
These supplement the canonical test/build commands, rather than replacing them.
Run shell scripts with Bash; countdowns also require authenticated `gh` and `jq`.
Notebook execution requires the repository's uv environment and notebook group.

- `countdown.sh owner/repo:pr ...` reports changes in pending PR checks.
- `runcountdown.sh owner/repo:run ...` reports workflow progress and approval waits.
- `notebook-env.sh [tutorial paper Dist]` executes PySnpTools notebook copies
  with a published FaST-LMM version (`FASTLMM_VERSION`, default `1.0.0`).
- `fastlmm-notebooks.sh [name ...]` executes FaST-LMM notebook copies under
  `tests/notebooks.toml`, reporting every policy exclusion. Both runners exit
  unsuccessfully if any requested execution fails.
- `compare-notebook-outputs.py REPO RUN_DIR NAME ...` prints text differences
  after normalizing timings and memory addresses. It is a review aid, not a
  numerical acceptance test; review images and stderr separately.
- `refresh-notebook-outputs.py REPO RUN_DIR` explicitly writes the reviewed
  FaST-LMM outputs back, preserving line endings and unexecuted prefix suffixes.
  It also retains the original helper's SingleSnpScale source portability edits
  and filtering of verbose file-I/O stdout. Review the resulting diff.

Set `REPO` (FaST-LMM) or `PST` (PySnpTools) to override checkout locations;
set `ENV_DIR` and `RUN_DIR` for isolated environments and outputs. Defaults live
under `~/.cache/fastlmm-release-2026`, outside the checkout. Use `RUN_DIR` under
`/mnt/c/deldir/fastlmm-release-2026` when outputs should be accessible on Windows.
The notebook runners use local source for the package being tested and are
development checks; installed-artifact qualification remains a separate gate.

The countdowns report completion and failure names but can exit successfully
when a completed run failed. Always inspect `CI result` and the exact PR head
before merging. Preserve the original scratch copies until these retained
versions have been accepted.
