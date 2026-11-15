<!-- TODO0 consider deleting this spec once the work below is implemented and released. -->

# GPU and float32 support for `single_snp`

## Goal

Make `single_snp(xp="cupy")` work again, add a `dtype` option (`"float64"` default, `"float32"`),
and measure whether GPU float32 is worth recommending.

## Background

- FaST-LMM computes in float64 because PySnpTools' `SnpReader.read` defaults to `dtype=np.float64`.
  Nothing in `single_snp` chose the precision explicitly.
- Consumer NVIDIA GPUs run float64 at 1/64 of the float32 rate (RTX 3060: a 4000² matmul is 43×
  faster in float32). Earlier GPU benchmarks were float64 and ran at roughly 32-thread CPU speed.
- On CPU (OpenBLAS), float32 `eigh`/SVD is no faster than float64; only matmul gains about 2×.

## Decisions

- **Device and precision are independent.** `xp` picks the device and `dtype` the precision, so the
  2×2 grid separates device effects from number-format effects, and float32 accuracy can be
  tested in CI without a GPU.
- **`xp` keeps its `ARRAY_MODULE` mechanism.** It lets a multiprocess runner put some tasks on the GPU
  and others on the CPU (`taskindex_to_environ`). `dtype` is applied by explicit casts where
  arrays enter the math: covariates, phenotype, kernel (`G` or `K`), and each test-SNP block.
- **Mixed precision falls out naturally.** With float32 inputs, eigendecomposition/SVD and SNP
  rotations run in float32; `h2` from the optimizer is a float64 array, which promotes the per-SNP
  statistics (O(N·S)) to float64.
- **PySnpTools fixes** (branch `cupy-sub-matrix`), needed for `xp="cupy"`:
  - `sub_matrix` subsets CuPy arrays instead of passing them to the numpy-only Rust helper.
  - `PstData` keeps float32 when moving a numpy array to CuPy instead of silently upcasting.
  FaST-LMM's GPU path therefore needs a PySnpTools release containing these fixes, followed by a
  raised lower bound in `pyproject.toml`.

## Changes in FaST-LMM (branch `gpu-f32`)

- `single_snp`: new `dtype` parameter (last keyword), validated; numpy mask for phenotype filtering;
  `EffectSize` converted to numpy (broke the GPU path when added).
- `lmm_cov`: `h2` stays on the host and is moved to the device only where combined with `S`;
  `denom` and result buffers use `self._xp`; the h2 optimizer receives host values.
- Cache: the "UUY is None" placeholder is detected as a bool array (the old `!= "float64"` test
  silently dropped a float32 `UUY`), and a cache made with a different dtype raises `ValueError`.

## Results (2026-11-15, Ryzen 9 7950X + RTX 3060, end-to-end `single_snp`)

Full detail and conclusions are in `specs/experiments/gpu_f32.ipynb`. Speedup versus CPU float64
(32 threads), full rank / low rank:

| N | CPU f32 | GPU f64 | GPU f32 |
|---|---|---|---|
| 1,000 | 0.6× / 1.2× | 1.9× / 0.8× | 2.8× / 1.3× |
| 4,000 | 1.3× / 1.0× | 1.8× / 0.7× | 8.0× / 3.8× |
| 10,000 | 1.1× / 1.1× | 1.4× / 0.6× | 8.7× / 4.8× (14 s vs 123 s full rank) |

- Accuracy: max |Δlog10 p| about 1e-4 or less; top-20 SNPs and all p < 5e-8 calls unchanged.
  Hard case with h2 pinned at 0.99, GPU f32 full rank: max 1.3e-2, but only on SNPs with
  p ≈ 1e-164 (relative error about 1e-4 in log p); no hit call changed.
- Profile at N=10k, GPU f32 full rank (15.6 s): 5.5 s is the kernel `K = G Gᵀ` built in float64 on the
  GPU by PySnpTools (0.13 s in float32); 3.9 s cuSOLVER `eigh`; about 3.5 s SNP reads, host-to-device
  copies, and the float64 re-read for `EffectSize`.

## Validation performed

- New FaST-LMM tests (`test_dtype_*`, `test_cupy_matches_numpy`) pass in the GPU venv; tolerances set to
  about 20× measured differences.
- FaST-LMM CPU regression subset (project venv, published PySnpTools 1.0.1): 195 passed, 1 skipped
  (cupy test, no cupy in that venv), 1 failed: `test_old_sel_plus_pc`, because the legacy `fastlmmc`
  executable exits with SIGILL (status 132) on this AMD CPU. That test does not touch the changed code.
- PySnpTools legacy suite (`tests/test.py`) on `cupy-sub-matrix`: 744 tests OK, 3 skipped (the new cupy
  tests; no cupy in that venv). The new cupy tests pass in the GPU venv and fail without the fixes.

## Remaining work

- Build the kernel in the run's dtype (about 12× instead of 8.7× at N=10k). Needs a `dtype` argument on
  PySnpTools' private `SnpKernel._read_with_standardizing`, which would make FaST-LMM's float32 mode
  (not only the GPU path) require the new PySnpTools release. Decision for the maintainer.
- Consider: `_multi_compute_stats` re-reads every test-SNP block in float64 just to compute
  `EffectSize` variance.
- Consider: `lmm_cov.setSU_fromG` drops singular values below the absolute threshold `1e-10`; in
  float32, roundoff singular values are about 1e-6 relative, so near-null directions are kept.
  Measured effect is small, but a dtype-relative threshold may be more principled.
- The legacy `tests/test.py` path and the two-kernel (`K1`) mixing search have not been tried with
  `dtype="float32"` (`K1` forces numpy; the mixing search asserts numpy).
- Release PySnpTools with the fixes, raise FaST-LMM's lower bound, and document the GPU workflow
  (including the CuPy install: for example, `cupy-cuda13x[ctk]`).
