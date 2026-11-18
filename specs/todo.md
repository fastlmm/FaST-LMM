todo000 This is the main todo. Only the human can remove this opening instruction. Agents may create, update, and remove the other todo entries.

# Next

Items are ordered by what to do next. Agents may maintain the task list, moving completed work to the Done section and removing its original entry.

Implement the [public API backend and precision policy](GPU_API_POLICY_PROMPT.md): single-kernel single_snp supports CPU/GPU × float32/float64; other workflows must clearly report unsupported requests or fallback behavior. Use the documented API list, not just the notebook.

Investigate and correct numerical correctness issues before making performance claims. Preserve existing tolerances unless a separately justified accuracy decision is approved; do not widen them merely to pass.

- Investigate CuPy float64 independently fitted h2 variation. The earlier strict comparison failed, while holding CPU-fitted h2 fixed passed. Preserve both results.
- Investigate float32 epistasis cancellation near P=1. NumPy and CuPy float32 matched each other but exceeded the legacy fixture tolerance.
- Assess the mouse CuPy float32 maximum absolute delta log10(P) of 2.27e-4 against justified accuracy criteria; matching the top ten associations is insufficient by itself.
- Add regression coverage for any corrections and record unresolved failures explicitly.
- Evidence: [notebook report](/mnt/c/deldir/fastlmm-notebook-validation-20261117/REPORT.md), [mouse four-way report](/mnt/c/deldir/fastlmm-mouse-four-way-20261118/REPORT.md).

Carl needs to review all GPU alpha code and documentation changes across bed-reader, PySnpTools, and FaST-LMM to some degree before the next release.

- Prepare a linked inventory of changed files and logical review batches, including changes already published in the first alpha and subsequent development fixes. Record the comparison base and reviewed commit for each repository.
- Review numerical algorithms, precision, backend boundaries, memory ownership, public APIs, and thread control in depth. Review tests, packaging, dependencies, CI, and remaining code at a level appropriate to their risk.
- Review user guides, API documentation, notebook source and saved outputs, installation instructions, platform limitations, warnings, and performance claims for accuracy and consistency.
- Track Carl's review progress, questions, and required corrections by batch. Agent review and passing tests do not mark Carl's review complete; re-review subsequent changes as needed.
- Starting points: [bed-reader](/home/carlk/programs/bed-reader), [PySnpTools](/home/carlk/programs/PySnpTools), [FaST-LMM](/home/carlk/programs/FaST-LMM), [GPU documentation review](GPU_DOCS_REVIEW.md), and [release checklist](GPU_ALPHA_RELEASE_CHECKLIST.md).

Run the current integrated main notebook completely in all four NumPy/CuPy × float32/float64 combinations after any correctness fixes.

- Execute [the main notebook](../doc/ipynb/FaST-LMM.ipynb) with fresh kernels and sequential configurations. Preserve the source notebook's existing edits and saved outputs; save executed copies separately.
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
