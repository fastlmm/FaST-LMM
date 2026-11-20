<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

# GPU API migration

The decisions below consolidate Claude Code session
`d20939e8-608b-4a92-9888-929009e5019e` (November 16, 2026).

## Agreed API

- FaST-LMM retains `single_snp(xp=..., dtype=...)`.
- bed-reader exposes `open_bed.read(..., xp=None)` and `has_gpu_support()`.
- PySnpTools adds `xp=None` to `SnpReader.read`, `read_kernel`, and
  `KernelReader.read`. Explicit `xp` wins over `ARRAY_MODULE`; NumPy is the
  default when neither is supplied. Missing CuPy or native GPU support raises.
- Standardization and in-memory selection follow the existing array. They do
  not move it in response to `ARRAY_MODULE`.
- Ordinary GPU reads return CuPy arrays sharing Rust-owned CubeCL memory.
  User-held arrays use CuPy standardization and selection.
- Internal `SnpReader._read_with_standardizing` keeps BED reading and
  Unit/Beta standardization inside CubeCL, before export to CuPy. Nested BED
  subsets forward indexes, including order and duplicate selections. No owner
  detection, tags, speculative copy branch, or foreign-memory adoption is needed.
- PySnpTools builds kernels in blocks through that helper. FaST-LMM calls the
  SNP and kernel helpers rather than maintaining its own native GPU pipeline.
- Rust shares `ReadOptions<TVal>` and `ReadOptionsBuilder<TVal>` between CPU,
  cloud, and GPU reads, with the same CPU signatures as 1.2. With `cubecl-cuda`,
  `.read_cubecl(&mut bed)` returns `gpu::CubeClMatrix<TVal>`;
  `Bed::read_cubecl_with_options` accepts built options. GPU calls return
  `Box<gpu::Error>`, whose `Bed` variant wraps ordinary read errors without
  changing the seven-variant `BedErrorPlus` from 1.2. The `gpu` module compiles
  only with `cubecl-cuda`; docs.rs marks the module and both read methods with
  feature badges. GPU supports `i8`, `f32`, and `f64`; the wrapper exposes its
  CubeCL client and tensor handle. Standardization and its invalid-variance
  diagnostic remain internal; `read_and_fill` remains CPU-only.
- CPU-only builds support Rust 1.85; GPU support needs Rust 1.95, enforced by
  CubeCL. The project uses MSRV-aware dependency resolution and a tested CPU
  lockfile. One ndarray 0.16.1 is shared with numpy; CubeCL pins remain unchanged.
- Cargo feature `cubecl-cuda` selects the beta CUDA runtime. Python GPU wheels
  target Linux x86-64 and Windows x86-64. HIP and workshop examples remain deferred.

## Implementation invariants

- CubeCL standardization occurs before any CuPy operation sees the buffer.
  Export waits for native GPU work. CuPy views retain the native owner.
- The Python GPU owner waits for consumers on all CUDA streams before returning
  memory to CubeCL. Blocked kernel creation also finishes CuPy multiplication
  before releasing a block and releases unused CubeCL pool memory before returning
  a kernel to the eigensolver.
- General CuPy standardization uses per-SNP reduction workspaces, with no
  matrix-sized masks or copies. Centered float64 variance reductions support
  dosages without subtracting nearly equal squared sums. CubeCL remains exact
  for genotype data within the CPU implementation's exact-sum range.
- GPU matrices exceeding u32 indexing limits fail before allocation. Integer
  standardization sums have additional explicit limits.
- FaST-LMM preserves its existing CPU float64 kernel formation before the LMM
  converts to the requested dtype. Forming that kernel directly in float32
  regressed the existing large-phenotype EffectSize test. Public PySnpTools
  requests still return the requested dtype; GPU kernels use the run dtype.
- CubeCL also accepts general fractional floating-point arrays. Genotype columns
  use validated integer statistics; other columns use shifted, centered f64
  statistics on the GPU, with no full-matrix host transfer. Their rounded results
  need not be bit-identical to the CPU's cancellation-prone raw-moment formula.
  Nonrepresentable moments raise rather than being treated as constant columns.
- Native standardization waits for previously exported CuPy writes before
  reading, and for CubeCL writes before returning to existing CuPy views.
- Packed uploads on the external CUDA stream wait for CubeCL's prior work before
  overwriting a pooled allocation, then complete the upload before CubeCL reads
  it. Pool reuse is ordered only on CubeCL's own stream until that explicit wait.
- EffectSize uses Unit's trained standard deviation squared in float64, with
  infinity mapped to zero for constant SNPs. Float32 statistics carry float32
  rounding; float64 genotype statistics agree with direct raw variance to
  rounding, without reading genotype blocks a second time.

## Workflow backend and worker boundaries

`single_snp` resolves its public `xp` argument once and temporarily establishes
`ARRAY_MODULE`. Internal process entry points resolve the worker environment;
standalone lower-level APIs retain explicit `xp` overrides. Model construction
checks agreement with the worker environment and verifies the kernel array's
actual module. SNP testing verifies the actual read array and model backend.

A decomposed parent LMM must not determine a worker's backend. Runner payloads
carry host model arrays; each worker recreates the cached model and regression
state on its selected backend without repeating decomposition. Additional
phenotype workers use the same rule. Returned phenotype arrays are transferred
back to the reducer's backend. Normal runs without a runner retain their arrays.

The environment context is protected by a process-local reentrant lock: overlapping
`single_snp` calls in one process are serialized. Environment restoration occurs
on success and exceptions, including worker-process failure. This does not change
standalone readers' explicit override behavior.

Tests observe native GPU reads, CuPy eigendecomposition, actual SNP arrays, and
LMM evaluation in real mixed CPU/GPU subprocesses, for inner SNP and outer
chromosome scheduling and multiple phenotypes. Separate tests exercise concurrent
call serialization and environment restoration after both local and worker errors.

## Release qualification still required

These are coordinated development changes, not a published-artifact acceptance.
Before releasing, publish the prerequisite bed-reader and PySnpTools versions,
raise honest dependency bounds, and repeat final acceptance against those
published artifacts. Do not publish packages or approve deployments automatically.

Qualify the supported Python/OS matrix, HIP on AMD hardware, near-capacity GPU
pool interactions, and large matrices on a GPU with more than 16 GiB. The local
CUDA tests cannot establish those claims.

## Local validation (November 16, 2026)

The development and artifact environments use CPython 3.13 on Linux/WSL with an
NVIDIA RTX 3060. These results do not qualify other platforms or AMD hardware.

- Initial API migration: canonical PySnpTools suite, 457 tests, 11 existing
  optional-dependency skips; full single_snp suite, 68 passed and 3 HDF5 skips.
- Fractional-input compatibility fix: 42 independent native/CuPy/CPU regressions,
  including C/F order, both float dtypes, trained constants, missing values,
  Beta validation, and a large common offset. The negative-input regression was
  confirmed to fail on the old installed GPU implementation.
- Rust GPU suite: 6 passed, including established bitwise genotype comparisons;
  CPU Rust Unit/Beta tests: 2 passed. No new Clippy diagnostics in GPU code.
- Environment/worker fix: 11 real-process/concurrency/error-restoration tests
  passed. Existing dtype, CuPy, multiple-phenotype and runner regressions:
  20 passed. Explicit reader overrides retain separate standalone tests.
- Installed wheel qualification outside source checkouts: 53 compatibility and
  environment tests passed. The bed-reader wheel was also built from its source
  distribution, including the new nested general-statistics kernel module.
- Optional HDF5/BGEN dependencies were installed in the artifact environment;
  its optional-dependency checks passed (7 tests and 7 subtests).
- Backend naming and upload-ordering fix: CPU-only tests passed (29 unit,
  58 integration, 166 doctests); the renamed CUDA backend passed all 7 Rust GPU
  tests and 7 backend doctests. All 75 installed-wheel Python GPU compatibility
  tests and 11 FaST-LMM worker/environment tests passed. The new fused-read
  regression fails on the preserved pre-fix wheel and passes with the explicit
  CubeCL-to-upload stream completion wait. HIP compiles, but AMD runtime testing
  remains unavailable locally. CUDA's memory checker is unavailable with the
  current WSL driver; the final tests use normal asynchronous execution.
- The notebook's complete execution is saved under
  `~/.cache/bed-reader-gpu/qualification/gpu_fastest_path.executed.ipynb`.

The artifacts retain development versions; they are not published release
candidates. Dependency bounds and published-artifact acceptance remain part of
coordinated release preparation.
