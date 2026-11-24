# Synthetic FaST-LMM benchmark

## Result (November 22, 2026)

The dataset is `/home/carlk/.cache/synthetic-bed/synth_14925x100000.bed` (14,925 individuals x 100,000 SNPs, synthetic, nothing from WTCCC), with its matching `.pheno.txt`. The generator is `/mnt/c/deldir/synthetic-bed/make_synthetic_bed.py`. `single_snp` used one kernel built from all 100,000 SNPs, which were also the test SNPs.

Code: FaST-LMM `implement/cpu-f32-eigh` at `a4d7580` (CPU float32 builds its kernel in float32 and decomposes it with SciPy's float32 LAPACK `evd` driver), PySnpTools `cupy-sub-matrix` at `b4b57d4`, and CPU-only bed-reader `implement/cpu-only` at `53e2eb5`. Machine: Ryzen 9 7950X (16 cores, 32 threads), RTX 3060, otherwise idle. Three timed repeats per configuration after one warm-up, each in a fresh process (one repeat for CPU float64 at 32 threads). Results: `~/.cache/synthetic-bed/benchmark-results/2026-11-22-cpu-f32-eigh/` (`summary.json`; summary script `/mnt/c/deldir/synthetic-benchmark/summarize_cpu_f32_eigh.py`).

Median seconds per stage:

| Configuration | Read | Copy | Kernel | Eigendecomposition | h2 fit | Association | Total (range) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU float32 | 5.5 | 2.1 | 4.7 | 12.4 | 0.4 | 7.2 | 33.2 (32.6–33.6) |
| CPU float32, 8 threads | 3.7 | — | 26.7 | 84.4 | 3.0 | 69.3 | 187.9 (185.0–190.0) |
| CPU float32, 16 threads | 3.7 | — | 19.7 | 77.3 | 3.4 | 53.9 | 158.9 (158.2–162.2) |
| CPU float32, 32 threads | 3.8 | — | 17.2 | 112.3 | 3.5 | 51.0 | 187.5 (173.7–208.3) |
| CPU float64, 16 threads | 4.0 | — | 38.1 | 170.9 | 64.8 | 113.1 | 395.8 (387.5–399.0) |
| CPU float64, 32 threads (one run) | 4.2 | — | 31.3 | 266.3 | 68.3 | 120.6 | 492.2 |

Findings:

- 16 threads is the fastest CPU setting for both precisions; 32 threads (all logical CPUs) is slower and noisier, mainly in the eigendecomposition.
- At 16 threads, CPU float32 is 2.5x faster than CPU float64 (158.9 s against 395.8 s).
- GPU float32 is 4.8x faster than the fastest CPU float32 and 11.9x faster than the fastest CPU float64.
- Accuracy against CPU float64 (16 threads): GPU float32 max absolute log10(P) difference 3.42e-3, h2 0.490116 against 0.490201; CPU float32 1.44e-3, h2 0.490230. CPU and GPU float32 differ from each other by up to 3.30e-3. All configurations rank the top 10 SNPs identically. CPU float64 at 32 against 16 threads differs by 4.4e-10.

Earlier result (November 21, 32 threads, before the float32 kernel and decomposition fixes): CPU float32 384.9 s, GPU float32 30.3 s, CPU float64 about 420 s (one run). That CPU float32 path built its kernel in float64, and NumPy's `linalg.eigh` decomposed it in float64, so it was not a real float32 computation.

## Rerunning

Use an isolated venv with editable installs of the three repositories and CuPy,
then run the driver with a new `--output` directory so every configuration is
measured again. The driver uses its own Python interpreter for workers by default.

```bash
python -u tools/synthetic-benchmark/driver_synthetic.py --repeats 3 --output ~/.cache/synthetic-bed/benchmark-results/2026-11-24-development
```

The driver runs one worker at a time, waits for the GPU to be idle, and reuses
completed JSON results only when the recorded checkout revisions, interpreter,
and input paths match. Use a new output directory if uncommitted code or input
files change. A stopped repeat may leave a partial `.log` with no JSON and is
rerun when resumed.

