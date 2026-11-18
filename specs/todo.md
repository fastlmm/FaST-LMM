todo000 This is the main todo. Only the human can remove this opening instruction. Agents may create, update, and remove the other todo entries.

# Next

Items are ordered by what to do next. Agents may maintain the task list, moving completed work to the Done section and removing its original entry.

Validate and review the canonical combined API-policy and numerical changes before making performance claims. Preserve existing tolerances unless a separately justified accuracy decision is approved; do not widen them merely to pass.

- Review and commit the canonical combined change in `investigate/gpu-alpha-numerics`, including the reconciled `implement/gpu-api-policy` edits. Combined validation passes; see [combined status](GPU_API_POLICY_STATUS.md). Preserve the original numerical failures and accepted precision exceptions as evidence.
- Review the focused product-precision correction and its regression. The full 1,225-pair notebook epistasis workload now passes the unchanged checks; this is not full notebook qualification.
- Evidence: [integrated numerical record](GPU_NUMERICAL_CORRECTNESS.md), [detailed report](/mnt/c/deldir/fastlmm-correctness-20261118/REPORT.md), [earlier notebook report](/mnt/c/deldir/fastlmm-notebook-validation-20261117/REPORT.md), and [earlier mouse report](/mnt/c/deldir/fastlmm-mouse-four-way-20261118/REPORT.md).

Carl needs to review all GPU alpha code and documentation changes across bed-reader, PySnpTools, and FaST-LMM to some degree before the next release.

- Prepare a linked inventory of changed files and logical review batches, including changes already published in the first alpha and subsequent development fixes. Record the comparison base and reviewed commit for each repository.
- Review numerical algorithms, precision, backend boundaries, memory ownership, public APIs, and thread control in depth. Review tests, packaging, dependencies, CI, and remaining code at a level appropriate to their risk.
- Review user guides, API documentation, notebook source and saved outputs, installation instructions, platform limitations, warnings, and performance claims for accuracy and consistency.
- Track Carl's review progress, questions, and required corrections by batch. Agent review and passing tests do not mark Carl's review complete; re-review subsequent changes as needed.
- Starting points: [bed-reader](/home/carlk/programs/bed-reader), [PySnpTools](/home/carlk/programs/PySnpTools), [FaST-LMM](/home/carlk/programs/FaST-LMM), [GPU documentation review](GPU_DOCS_REVIEW.md), and [release checklist](GPU_ALPHA_RELEASE_CHECKLIST.md).

Finish qualification of the four completed notebook configurations against the canonical combined source; rerun affected configurations only as source/provenance review requires.

- The stopped API-policy chat completed all four fresh, sequential configurations under `/mnt/c/deldir/fastlmm-api-policy-20261118/notebooks-final/`. Preserve those executed copies and per-run source manifests, including `numpy-float64-final`. Compare their full numerical outputs before declaring qualification passed; reconcile differences from the final canonical source. Preserve the main notebook's existing edits and saved outputs.
- Verify numerical results, identities, ordering, finite masks, warnings, and explicit failures. Record exact commits, installed artifacts, import paths, dependencies, thread settings, and per-cell timings.
- Existing combined validation recorded 157 passed, 44 subtests passed, and two runner doctests; it did not rerun the full integrated notebook.
- Reference: [post-alpha integration and release checklist](GPU_ALPHA_RELEASE_CHECKLIST.md#post-alpha-runner-and-notebook-integration).

Benchmark the larger mouse problem with 1, 4, and 16 CPU BLAS threads after correctness qualification.

- Run NumPy float32 and float64 with each thread count, verify actual loaded BLAS pool limits, and retain identical data and statistical settings.
- Use fresh processes, one full warm-up, and five timed repetitions per configuration. Synchronize GPU timing and retain first-call costs separately.
- Run configurations sequentially without competing notebook workloads. Compare GPU results with the fastest measured CPU configuration at the same precision.
- The earlier approximately 10× float32 speedup used one CPU thread; it does not establish the fastest CPU configuration.
- Evidence and dataset details: [mouse four-way report](/mnt/c/deldir/fastlmm-mouse-four-way-20261118/REPORT.md).

Reconcile notebook documentation and dependency requirements with the next PySnpTools alpha.

- The revised notebook uses num_threads, which published PySnpTools 1.1.0a1 does not support. Document the development requirement until a compatible release is available.
- Review dependency bounds, direct threadpoolctl requirements, installation instructions, GPU platform limits, and precision guidance together.
- Reference: [release checklist](GPU_ALPHA_RELEASE_CHECKLIST.md).

Prepare the next coordinated alpha and its artifact validation plan. Do not publish, push release tags, or merge into main without explicit human authorization.

- Prepare the updated PySnpTools alpha first. Determine whether bed-reader needs another release based on required changes rather than automatically bumping all three projects.
- Prepare FaST-LMM dependency bounds and release notes, then perform final acceptance against published prerequisites after authorized prerequisite publication.
- Build and test wheels and source distributions in isolated environments outside source checkouts. Cover supported Python boundaries, relevant optional dependencies, CPU platforms, and supported GPU configurations; report unavailable checks explicitly.
- Retain numerical exceptions and platform failures as visible release decisions. Do not reuse an existing published alpha version.

Maintain one qualification record linking exact commits, artifact hashes, environments, executed notebooks, numerical comparisons, benchmark results, and remaining release gates.

- Keep reusable scripts and run evidence under a findable folder in /mnt/c/deldir; keep virtual environments and large build output under ~/.cache or /mnt/c/deldir.
- Distinguish old worktree evidence from validation of the integrated development heads.
- Baseline heads verified on November 18, 2026: bed-reader gpu-cubecl at 71f62e4; PySnpTools cupy-sub-matrix at 75652ce; FaST-LMM gpu-bed-reader at d37dccf. All three primary checkouts were clean; existing worktrees were preserved.

# Done

- November 18, 2026: reproduced all three numerical findings against explicitly identified integrated development code in isolated branch `investigate/gpu-alpha-numerics`. Preserved the original worktrees, opening todo000 instruction, and notebook outputs.
- Corrected premature float32 epistasis product rounding, added a regression that fails before the fix, and retained all tolerances. Full notebook epistasis max absolute P error fell from `2.7787e-5` to `2.4410e-6`; corrected NumPy/CuPy defaults agree exactly.
- Diagnosed float64 fitted-h2 variation at the likelihood rounding floor and mouse GPU32 eigensolver precision. Documented both limitations and retained failed accuracy gates above. Float64 epistasis: 15 tests and two subtests pass; backend tests: 26 pass; 11 pre-existing float32 fixture failures remain explicit.
- Recorded exact commits, environment/import paths, development binary hash/source equivalence, commands, baseline/corrected comparisons, and controlled precision diagnostics in the [qualification report](/mnt/c/deldir/fastlmm-correctness-20261118/REPORT.md). No human review or release gate was marked complete.

- November 18, 2026: Carl accepted the measured mouse CuPy float32 precision tradeoff for the alpha, with a user warning and concrete example. Added the warning to the GPU guide and single_snp dtype documentation: CPU64 P=`2.0918834e-87` versus GPU32 P=`2.0899597e-87` at fixed h2 (about 0.092% relative error). Retained the strict comparison exceedances as an approved precision exception; no tolerance, expected output, solver, or default changed. This does not complete other numerical gates or Carl's cross-repository review.

- Follow-up: investigated the 11 additional epistasis failures. Ordinary float32 preprocessing/decomposition exceeded the unchanged `1e-5` fixtures; partial promotion was insufficient. Epistasis now explicitly computes on CPU float64 and warns on other defaults. Worker reads enforce float64 independently of worker defaults, and float32 eigen caches are rejected rather than widened. This qualifies the CPU64 execution path, not native float32 epistasis.
- Follow-up: checked float64 h2 with independent long-double likelihood derivatives. Ordinary fits differ from the reference optima by less than `4.982e-8` in h2, with likelihood loss below `3.948e-13` and maximum relative P error about `0.000031%`. Reference CPU/GPU fits agree to about `3e-13` in log10(P). Recommendation: retain the optimizer and documented fitting-roundoff limitation; preserve the original strict diagnostic exceedances without widening a tolerance. No additional human approval is implied.

- Canonical integration: combined the API backend/precision policy from `implement/gpu-api-policy` into `investigate/gpu-alpha-numerics`, preserving both dirty worktree snapshots under `/mnt/c/deldir/fastlmm-correctness-20261118/integration/`. Reused the shared CPU64 scope and workflow lock instead of duplicate epistasis policy logic; retained explicit worker reads, the product correction, strict cache validation, mouse warning, long-double h2 evidence, and all numerical regressions. Combined validation now passes; Carl's review remains open.

- Canonical integration validation: 227 canonical tests run with no failures; five optional HDF5 skips all passed separately against the final source. Combined Python 3.13 regressions passed 60 tests and 94 subtests. Python 3.11 passed 37 API/epistasis tests; Python 3.15 passed 35 with two supported-platform GPU skips. Strict docs and Ruff pass. Fresh synthetic/mouse/epistasis controls preserve the diagnosed/accepted limitations without changing tolerances.
- Preserved and reconciled the stopped API-policy worktree's final handoff and source edits. Retained the matched-h2 effect-size test, independent-fit P-value check, all product/cache/worker regressions, and the latest public-API coverage. No original worktree or generated evidence was deleted. The four completed notebook executions are linked above; their numerical cross-comparison remains the next qualification task.
