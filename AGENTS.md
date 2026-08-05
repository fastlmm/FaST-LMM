# Coding Notes for Agents

This file contains repository-wide guidance for FaST-LMM, a mature numerical
Python package with downstream users. Preserve compatibility and numerical
correctness unless the task explicitly changes those goals.

## General Policies

- When work is interrupted or reaches a stopping point, report the current
  status, what remains, and the recommended next step. If the next step is
  within the current task and safe to perform, perform it instead of merely
  recommending it.
- Inspect existing code, tests, documentation, and project configuration before
  introducing a new pattern. Prefer a focused change that follows nearby
  conventions over an unrelated cleanup or broad rewrite.
- Do not silently skip required Python versions, operating systems,
  architectures, dependency configurations, tests, doctests, or artifact
  checks. If a required environment or tool is unavailable, fail clearly and
  report what is missing.
- Avoid silent clamping, coercion, truncation, or fallback behavior. Validate
  inputs and fail clearly unless the public API explicitly documents another
  behavior.
- Preserve behavior on every supported Python version, not only the interpreter
  used for development.

## Python 3.14 Support

For work related to Python 3.14 support, read and follow
[`specs/PYTHON_3_14_SUPPORT_SPEC.md`](specs/PYTHON_3_14_SUPPORT_SPEC.md) before
making changes.

Keep the spec aligned with implementation decisions that materially change its
scope, dependency requirements, validation matrix, artifact requirements, or
release plan. Do not mark Python 3.14 support complete until the prerequisite
`pysnptools` and `fastlmmclib` releases and the spec's acceptance criteria are
satisfied.

## Numerical Correctness

- Treat changes to numerical algorithms, tolerances, array shapes, dtypes,
  missing-value handling, sample or marker ordering, and random-number behavior
  as behavior changes, not mechanical refactors.
- Do not update expected numerical or textual output merely to make a test pass.
  First confirm that the new result is correct and does not hide a regression.
- Use deterministic seeds in tests involving randomness. When exact equality is
  inappropriate, use an explicit, justified tolerance rather than an
  unnecessarily broad one.
- Preserve representative small end-to-end tests for association, inference,
  and packaged data so integration failures are not hidden by unit tests.

## Error Handling

- Preserve useful diagnostics and the original exception when translating
  errors. In Python, use exception chaining (`raise ... from error`) when adding
  context.
- Catch only the exceptions that can be handled meaningfully. Do not replace a
  useful exception with a generic message, return sentinel values for unexpected
  failures, or use a broad `except` to make a failing path appear successful.
- Do not ignore meaningful return values, warnings, or exceptions solely to
  quiet tests or CI. Fix the cause or document a narrowly justified exception.

## Dependencies and Packaging

- Treat dependency upgrades as behavior migrations, not just resolution or
  import fixes. Review changed defaults and APIs, then test affected behavior.
- Keep dependency-upgrade changes focused when practical. Avoid mixing them with
  broad code or documentation churn unless the additional changes are required
  by the migration.
- Declare direct runtime imports as direct project dependencies with honest
  lower bounds. Use Python-version markers when a newer interpreter needs a
  newer dependency without unnecessarily raising requirements elsewhere.
- Build and test both the wheel and source distribution in isolated
  environments. Verify that package metadata, licenses, sample data, executables,
  and other required package data are present.
- Test the installed artifact without relying on the repository being on
  `PYTHONPATH`; a passing test against the source checkout is not sufficient
  evidence that the package is correct.
- Do not add redundant wrappers or task aliases that merely duplicate the
  repository's canonical `uv`, test, lint, or build command.

## Generated and Vendored Files

- Treat generated files and built documentation as outputs, not sources of
  truth. Find and edit the corresponding source, template, or generator first.
- If an urgent change must touch generated output directly, make the matching
  source or generator change in the same work so regeneration will not revert
  it.
- Do not casually edit vendored JavaScript, bundled executables, binary test
  data, or third-party code. Explain and verify any task that genuinely requires
  such a change.
- Regenerate affected outputs and check that the resulting diff contains only
  intended changes.

## API and Code Design

- Preserve documented public APIs and serialized or file-format compatibility.
  Treat changes to public signatures, defaults, accepted input shapes or types,
  result schemas, import paths, and exceptions as compatibility-sensitive.
- Prefer one clear, canonical API path. Keep aliases or compatibility shims only
  when downstream compatibility requires them, and document the canonical path.
- Keep implementation details private. Do not expose helpers publicly merely to
  make internal code or tests convenient.
- Place functionality with the abstraction it belongs to. When moving an
  abstraction, account for its related helpers, tests, examples, and
  documentation instead of leaving partial duplicate implementations.
- Prefer concise names when module or class context already provides the missing
  context. Avoid obscure abbreviations, while retaining conventional scientific
  notation where it is established and unambiguous in the surrounding code.

## Tests and Validation

- Add or update tests for every behavior change and regression fix. A regression
  test should fail for the original defect and pass for the corrected behavior.
- Run the narrowest relevant tests while iterating, then run the repository's
  complete required validation appropriate to the change before handing work
  back.
- Do not disable, deselect, or weaken tests, doctests, warnings, lint rules, or
  CI jobs merely to obtain a passing result. Any necessary exception must be
  narrow, documented, and reported.
- Compare legacy `tests/test.py` coverage with pytest collection when changing
  test orchestration. Do not retire either path until the replacement is shown
  to cover the required unit tests, integration tests, and doctests.
- For packaging or dependency work, validate supported boundary Python versions
  and relevant optional-dependency configurations, not just the default local
  environment.

## Comments and Documentation

- Preserve useful TODOs, diagnostic notes, and debugging comparisons while
  their underlying issue remains unresolved. If a comment may be obsolete but
  that has not been established, annotate or report it rather than silently
  deleting it.
- Keep public documentation synchronized with signatures, defaults, exceptions,
  supported versions, installation steps, and examples.
- Prefer one complete, executable example for a public workflow and link to it
  from related documentation instead of maintaining multiple drifting copies.
- Use American English. In Markdown, place blank lines around headings, lists,
  and fenced code blocks, and keep list markers consistent.

## Development and Release Safety

- Use the repository's documented, project-local toolchain. Do not silently
  install system-wide packages or alter unrelated global configuration.
- Do not publish a real release to PyPI or another package index. Agents may
  prepare version changes, release notes, artifacts, and commands, but a person
  must perform or explicitly authorize the publication step.
- Ordinary CI and release qualification must not rely on unpublished sibling
  checkouts, unreviewed prereleases, or accidental local state.
- Before handing work back, summarize validation performed, anything not run,
  and any compatibility, numerical, packaging, or release risk that remains.
