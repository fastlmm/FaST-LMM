todo000 This is the main todo. Only the human can remove this opening instruction. Agents may create, update, and remove the other todo entries.

# Next

Items are ordered by what to do next. Agents may maintain the task list, moving completed work to the Done section and removing its original entry.

Every code review (bed-reader Rust and Python, PySnpTools, FaST-LMM) also checks safety and ownership:

- User-supplied indexes and sizes: negative, repeated, out-of-range, empty, and huge values; u32 and usize overflow; and whether every value that reaches a `launch_unchecked` kernel or raw pointer was validated first.
- Every `unsafe` block: it has a `SAFETY` comment, and the stated invariant really holds on every path, including error paths.
- Ownership: no double free, use after free, or leak on error paths; GPU memory outlives every CuPy view and `__cuda_array_interface__` pointer; cloned handles; `Drop` order; `Send` impls and mutexes across threads.
- Python inputs passed into Rust: wrong dtype, shape, strides, memory order, or device.

Review Rust bed-reader code (work branch `implement/read-cubecl-method`, based on `gpu-cubecl` at `49b670b`, with approved formatting commit `0b51112`), one batch at a time, in this order:

1. Public API: `src/gpu.rs` (182 lines: feature-gated module docs with the CPU/GPU read table, five-variant `Error`, boxed conversions, GPU re-exports including the `gpu::client` and `gpu::memory_cleanup` functions), the `src/lib.rs` changes (`ReadOptions<TVal>` and its shared builder; the three GPU reads `Bed::read_cubecl`, `ReadOptionsBuilder::read_cubecl`, and `Bed::read_cubecl_with_options`, cross-linked with their CPU counterparts and listed in Main Functions; docs.rs feature badges; unchanged `BedErrorPlus`), and `Cargo.toml` plus `.cargo/config.toml` (CPU MSRV 1.85, GPU Rust 1.95, `cubecl-cuda`, `_cubecl-cpu-test`, dependency resolution, docs.rs metadata, version 1.3.0-beta.1). Carl's earlier batch 1 review covered the typestate design and must be redone against the final API.
2. GPU matrix type, `src/gpu/reader.rs` lines 1–139, 311–486, and 499–530: the `client` and `memory_cleanup` functions, `CubeClVal`, `CubeClMatrix` (`dim`, `is_f`, `to_cpu`, `client`, `tensor_handle`, `cuda_pointer`), `CudaPointer`, size helpers (`to_u32`, `element_count`), dedicated allocations.
3. GPU read, `src/gpu/reader.rs` lines 140–310, 487–498, and 531–570: `read_cubecl` (index resolution, reading the file into the packed buffer, size checks, the unchecked launch), `read_packed`, and the `decode` kernel. Index resolution, the file-length check, and the genotype table are shared with the CPU and cloud readers in `src/lib.rs` (`resolve_index`, `try_div_4`, `set_up_two_bits_to_value`).
4. Raw CUDA upload, `src/gpu/reader.rs` lines 571–697: `upload_pinned_cuda` and `cuda_staging` (pinned buffers, streams, events, the retained context; six `unsafe` blocks).
5. GPU standardizer, `src/gpu/reader/standardize.rs` (450 lines): private standardizer `Error`, `snp_sums` (0/1/2/NaN detection, plane and tree reductions), per-SNP statistics, the `apply` kernel, and the plane-versus-tree test. Host slice traversal uses Rust-1.85-compatible chunks with unchanged numerical behavior.
6. Fractional-value statistics, `src/gpu/reader/standardize/general.rs` (182 lines): the `centered` kernel (parallel two-pass f64) and `mean_std`, whose invalid-variance diagnostic is private.
7. Python bindings, `src/python_module.rs` `gpu_python` (lines 818 to the end): GPU and private standardizer errors mapped to the unchanged Python exception types/messages, `GpuArray` ownership, `Drop` (memory release, unraisable errors), `__cuda_array_interface__`, `read_gpu`, `allocate_gpu`, and `standardize_gpu`. Also decide the CuPy device mismatch: reads always target device 0, so a caller who switched CuPy to another device (`cupy.cuda.Device(1).use()`) probably gets a CuPy device-mismatch error when combining arrays (unverified; one GPU here). Either raise a clear error when CuPy's current device is not 0 or document the limit.
8. Tests and CI: `src/gpu/reader.rs` tests (lines 698 to the end), the CPU builder inference and exhaustive-error regressions in `tests/tests_api.rs`, native Python index/variance exception regressions, CPU-runtime kernel tests, Rust 1.85 CPU tests, formatting enforcement, and GPU wheel jobs.

Batch 1 decisions and outcomes, decided by Carl on November 19, 2026:

- Decision 1: name the feature-gated device module `gpu`, preserving `CubeClMatrix`, `CubeClVal`, `CudaPointer`, and the terminal method names (`CubeCl` was later replaced; see the November 20 decisions). This avoids hiding the external `cubecl` crate; verify the separate name probe.
- Decision 2: isolate rustfmt changes and CI's formatting check in approved commit `0b51112` (three files, 104 insertions and 23 deletions), including rustfmt's ordering changes as explicitly approved. Earlier implementation work was backed up, stashed, and restored on top; all further work stays uncommitted pending Carl's review.
- Decision 3 / finding 1, resolved: GPU calls return `Box<gpu::Error>`; `Error::Bed` preserves ordinary read failures. `BedErrorPlus` matches main's seven variants exactly. Python GPU failures remain `RuntimeError`; wrapped read failures follow the existing index/I/O/value exception mapping. Exhaustive 1.2 matches compile with and without GPU support; semver checking reports zero breaks.
- Decision 4, resolved: compile `gpu` only with `cubecl-cuda`; add nightly docs.rs feature badges to the module, `read_cubecl`, and `read_cubecl_with_options`.
- Decision 5 / finding 3, resolved by Carl's explicit policy decision: keep the `cubecl-cuda` name and Beta wording, despite the general `_experimental` naming preference. Keep the CubeCL pins and exposed CubeCL types.
- Decision 6 / finding 5, resolved: restore the CPU MSRV to 1.85; GPU support needs 1.95 through CubeCL. Use resolver fallback, retain psm 0.1.28 and one ndarray 0.16.1, and select idna_adapter 1.2.0 so CPU URL handling uses ICU 1.x independently of optional GPU ICU 2.x requirements. Add locked CI tests on Rust 1.85; both CPU configurations pass locally with `--locked`.
- Decision 7 / finding 7, resolved: public `gpu::Error` contains `Bed`, `Unavailable`, `Server`, `TooLarge`, and `Cuda`, with no inner feature gates. Move `InvalidVariance` to the crate-private standardizer error while preserving its Python `RuntimeError` message. No GPU error items remain in CPU-only builds.
- Decision 8 / finding 8, resolved: remove the internal `cubecl` feature and its compile-error path. Keep `_cubecl-cpu-test`; Carl chose go on the experiment and explicitly approved removal of its two experiment comments. AGENTS.md explains that `--all-features` tests the CPU runtime, while `cubecl-cuda` tests CUDA.
- Decision 9, resolved: separate the new Bed method with a blank line, describe terminal read methods above the options table, remove them from the table itself, and remove the doubled blank line here. Finding 9 remains resolved: dropped dedicated memory returns at CubeCL's next allocation or explicit cleanup, consistently documented.
- Findings 2 and 4 remain resolved by shared option types and the separate `read_cubecl` method; inference is covered by a regression and the external probe. Finding 6 remains resolved by the public `Bed::read_cubecl_with_options` consumer of built options.
- No kernel, numerical, tolerance, expected-output, or Python API change is part of these decisions. Passing tests do not mark Carl's batch 1 review complete.

Final follow-up validation (November 19, 2026): formatting (stable and Rust 1.95), all CI Clippy configurations plus default features, Rust default/no-default/CUDA tests and doctests, all eight CPU-runtime kernel tests, both locked Rust 1.85 CPU configurations, the Rust 1.95 all-target CUDA check, all three compatibility probes, and the module-name probe passed. Semver against 1.2.1 passed all 196 applicable checks with zero breaks. Plain docs have 69 existing redundant-link warnings; current nightly docs have 71, with feature badges verified on all three requested items and no new warnings. Stable's two unused-dependency warnings were reproduced on original source and not suppressed. Both GPU wheel builds passed; the installed sdist-built wheel passed 239 tests with the same three EBI skips (116 GPU/API tests, five new exception cases), PySnpTools passed 104 plus seven subtests, and FaST-LMM passed 61 plus 82 subtests. Local runtime validation used Linux/WSL, Python 3.13, and an RTX 3060. Other platforms and Python versions were not rerun; no push, CI dispatch, or further commit was authorized or performed.

Batch 1 decisions and outcomes, decided by Carl on November 20, 2026 (goal: the GPU API reads as if designed from the start, with no leftovers from earlier designs):

- Replace the empty `CubeCl` struct, a leftover typestate marker, with the functions `gpu::client()` and `gpu::memory_cleanup()`. Python's bindings call them internally; Python behavior is unchanged.
- GPU reads are exactly three, each the counterpart of a CPU read of a local file: `Bed::read_cubecl` (of `Bed::read`), `ReadOptionsBuilder::read_cubecl` (of `ReadOptionsBuilder::read`), and `Bed::read_cubecl_with_options` (of `Bed::read_with_options`). `Bed::read_cubecl` is new. Each is reachable from the top of the docs (Main Functions, the ReadOptions section, README example, and the `gpu` module's CPU/GPU table), names its CPU counterpart, and links both ways.
- Removed wording that described changes rather than the design ("leaves CPU option types unchanged", "`BedErrorPlus` is unchanged", "stays CPU-only"). Replaced the typestate-era test `gpu_builder_chain_order` with `gpu_entry_points_match_cpu`, which checks all three GPU reads against the CPU read.
- Documented device 0 with a recipe: on a machine with several GPUs, choose one with `CUDA_VISIBLE_DEVICES`, or run one process per GPU.
- Deferred, each addable later without breaking anything:
  - GPU read-and-fill (three methods plus a public `CubeClMatrix::empty`): revisit if measured per-block GPU allocation cost justifies it.
  - Choosing a GPU device: revisit when a real user needs several GPUs in one process.
  - GPU reads from the cloud: GPU reads are local-only by design.
- Declined: uploading an `ndarray` into a `CubeClMatrix`, writing from a `CubeClMatrix` (use `to_cpu` then `write`), a public Rust GPU standardizer (the CPU standardizer is not public either), and a `BedCloud`-like GPU type (a GPU read has the same source, a local `Bed`; only the destination differs).
- After this work is committed: a separate commit converting the 71 existing redundant explicit rustdoc links in `lib.rs` to intra-doc links.

November 20 validation: formatting and all three strict CI Clippy configurations passed. Rust CUDA tests passed (38 unit, 120 integration with the two existing EBI ignores, 182 doctests); CPU-runtime kernels 8 passed; default (29, 120, 174 doctests) and no-default-features (29, 60, 167) passed; both Rust 1.85 CPU configurations passed with `--locked`; the Rust 1.95 all-target CUDA check passed. Nightly docs.rs-style and plain docs, with and without `cubecl-cuda`, have only the 71 existing warnings. A rebuilt GPU wheel passed bed-reader's installed suite (239 passed, three existing EBI skips), PySnpTools (104 plus seven subtests), and FaST-LMM (61 plus 82 subtests). No commit or push.

The prior implementation validation is preserved in [the earlier record](/mnt/c/deldir/bed-reader-read-cubecl/VALIDATION.txt). Final follow-up commands, lockfile versions, results, and remaining limits are recorded in [the follow-up record](/mnt/c/deldir/bed-reader-gpu-followup/VALIDATION.txt).

Rust review progress (November 20, 2026). For each batch, the agent reviews and fixes first; Carl then reads the cleaned-up code, focusing on judgment calls.

- Branch `implement/read-cubecl-method` commits: `c715daf` (the final GPU API), `05f5c15` (rustdoc link cleanup), `302a462` (batch 1 and 2 fixes), `9c1ecbe` (batch 3 fixes). CI on `05f5c15` passed all 74 jobs after rerunning two network-timeout failures; CI on `302a462` passed all 74 jobs; CI on `9c1ecbe` is running.
- Batch 1 (public API): agent review and fixes done and committed; Carl read the docs and approved them. Batch 2 (GPU matrix type): agent review and fixes done and committed; Carl read the docs and approved them. Batch 3 (GPU read): agent review and fixes done and committed in `9c1ecbe`; Carl reviewed the overview and asked for the sharing below. Batch 3 also fixed an old CPU and cloud bug: on mode 0 files, index errors named the wrong axis. CPU, cloud, and GPU reads now share one index resolver. Next: batch 4.

Things to remember:

- GPU reads need only the NVIDIA driver, not the CUDA Toolkit. CubeCL 0.11 compiles kernels with its bundled LLVM; NVRTC is used only by CubeCL's optional `cpp` feature, which bed-reader does not enable. Verified November 20 by hiding the toolkit's libraries (the read still worked) and the driver's (a clean `gpu::Error::Unavailable`, no panic). If bed-reader enables `cpp`, or a CubeCL upgrade changes its default backend, NVRTC and the toolkit become runtime requirements again; update the `gpu` docs.
- Python review (batch 7): check whether the Python GPU docs say a CUDA Toolkit is needed. bed-reader itself does not need one; CuPy has its own requirements.
- Rust 1.85 for downstream users needs Cargo's MSRV-aware resolver plus `cargo update -p psm --precise 0.1.28` (documented in README-rust.md). The cause is `ar_archive_writer` 0.5.1, a build dependency of `psm` (through `ndarray-npy`, `py_literal`, and `pest`), which uses Rust 1.88 syntax without declaring a `rust-version`. Revisit if upstream fixes its metadata (consider reporting it) or if the MSRV is raised.
- CubeCL turns on `cudarc`'s build-time `nvcc` check, so any installed `nvcc` newer than `cudarc` supports breaks GPU builds (`cudarc` 0.19.10 supports up to CUDA 13.4). After each new CUDA release, check that a `cudarc` release supports it. The `gpu` docs give the workaround.
- The documentation CI job runs Sphinx with `-W`, so a timeout fetching an intersphinx inventory (for example, `docs.scipy.org`) fails the job. Decide whether to make that robust.
- Upgrading CubeCL changes public types (`Client`, `TensorHandle`, and the errors in `gpu::Error`). With the GPU API labeled Beta rather than `_experimental`, each upgrade after 1.3.0 is a deliberate semver exception to call out in the release notes.

Review Python bed-reader code: `bed_reader/_gpu.py`, the GPU parts of `bed_reader/_open_bed.py`, `bed_reader/_array_module.py`, and `bed_reader/__init__.py`.

Paused November 18, 2026. bed-reader Python and Rust docs are reviewed and committed (`4fa5dcd`). The agent's Rust code pre-review fixes are committed in bed-reader: checked packed-input u32 indexing, `Drop` reports CUDA errors as unraisable instead of aborting Python, GPU failures raise `RuntimeError`, the CUDA context is retained once, dead code removed, and fractional-value standardization is a parallel two-pass f64 reduction (about 6–22 times faster, identical float32 results). All GPU tests passed (Rust 8, bed-reader 111, PySnpTools 104, FaST-LMM 61).

- GPU read and standardization tradeoffs (where read time goes, allocation cost, decode kernel, possible speedups): [ALGORITHM_TRADEOFFS.md](/mnt/c/deldir/bed-reader-gpu-memory/ALGORITHM_TRADEOFFS.md).
- PySnpTools no longer calls `gpu_memory_cleanup`, so its bed-reader lower bound must rise to the beta that returns memory on drop before release.

Validate and review the canonical combined API-policy and numerical changes before making performance claims. Preserve existing tolerances unless a separately justified accuracy decision is approved; do not widen them merely to pass.

- Review the committed canonical combined change in `investigate/gpu-alpha-numerics`, including the reconciled `implement/gpu-api-policy` edits. Combined validation passes; see [combined status](GPU_API_POLICY_STATUS.md). Preserve the original numerical failures and accepted precision exceptions as evidence.
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

- November 19, 2026: decided to keep f64 accumulation for GPU fractional-value standardization statistics. f32 would save about 10–25 ms on a 250,000 x 1,000 float32 block (17 ms versus 27–42 ms) but has about 5 times the error (about 2e-7 versus 4e-8). Scripts: `/mnt/c/deldir/bed-reader-gpu-memory/accuracy_fractional.py` and `bench_fractional.py`.
- November 19, 2026: bed-reader CI builds GPU wheels for Linux x86_64 (manylinux_2_34, alongside CPU-only manylinux2014 wheels for older glibc) and Windows x64; other platforms stay CPU-only, and every installed wheel is checked for the expected GPU support (`40bc892`, `49b670b`, CI 73 of 73). The CI-built Linux wheel passed 111 GPU tests on WSL2 with Python 3.13; native Linux and Windows GPU runs and a real old-glibc host are untested.
- November 19, 2026: bed-reader CI runs the CubeCL kernel tests on CubeCL's CPU runtime (internal `_cubecl-cpu-test` feature, `ee33633`); the `snp_sums` plane and tree reductions are tested against each other on CUDA. Rust CI lints `cubecl-cuda` only where it can build (`0001b2a`).
- November 19, 2026: PySnpTools runner thread tests accept the missing BLAS pool on macOS (Accelerate) and Windows ARM while still checking runner limits and environment variables; Linux and Windows x86 still require a pool. Commit `238aae1` on `cupy-sub-matrix`; CI passed 39 of 39 (run 44551444730).
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

- November 18, 2026: at Carl's explicit request, committed the original primary task list and [public API policy prompt](GPU_API_POLICY_PROMPT.md) in `b158928`, committed the qualified combined implementation in `2b11c66`, and integrated it into `gpu-bed-reader`. Preserved this opening todo000 instruction, all remaining qualification tasks, original evidence, and the read-only review snapshot. Merge authorization does not mark Carl's broader review complete; saved-notebook comparison remains next.
