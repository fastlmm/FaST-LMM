# Synthetic FaST-LMM benchmark

## Result (November 25, 2026, release candidate)

Same dataset and model as below. Code: FaST-LMM `gpu-bed-reader` at `14f68ac` and PySnpTools `cupy-sub-matrix` at `df052a3` (both editable checkouts), with the published bed-reader 1.3.0b3 wheel, NumPy 2.5.5, SciPy 1.18.1, and CuPy 14.2.0, in `~/.cache/fastlmm-bench-2026-11-25/venv`. Machine and protocol as below (Ryzen 9 7950X, RTX 3060, otherwise idle; three timed repeats after one warm-up, one repeat for CPU float64 at 32 threads). Command: `driver_synthetic.py --repeats 3 --prerequisites checkout`. Results: `~/.cache/synthetic-bed/benchmark-results/2026-11-25-dev-checkouts/` (`summary.json`; summary script `~/deldir/synthetic-benchmark/summarize.py`).

Median seconds per stage:

| Configuration | Read | Copy | Kernel | Eigendecomposition | h2 fit | Association | Total (range) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU float32 | 2.0 | 2.2 | 4.7 | 11.9 | 0.4 | 7.1 | 29.1 (29.0–30.2) |
| CPU float32, 8 threads | 0.9 | — | 27.8 | 81.3 | 3.3 | 69.9 | 185.0 (180.1–186.5) |
| CPU float32, 16 threads | 0.9 | — | 20.3 | 75.3 | 2.7 | 54.0 | 154.3 (152.3–154.7) |
| CPU float32, 32 threads | 1.1 | — | 16.3 | 130.0 | 3.3 | 48.8 | 200.4 (166.0–204.3) |
| CPU float64, 16 threads | 1.6 | — | 36.8 | 164.9 | 65.9 | 108.6 | 379.2 (375.0–381.3) |
| CPU float64, 32 threads (one run) | 2.1 | — | 31.6 | 249.2 | 66.5 | 109.8 | 460.6 |

Findings:

- GPU float32 is 13.0x faster than CPU float64 and 5.3x faster than CPU float32, both at 16 threads. CPU float32 is 2.5x faster than CPU float64. User-facing docs say over 10x and about 5x and do not quote times, except the benchmark page, which gives the measured times and 13x (Carl, November 25).
- 16 threads is again the fastest CPU setting for both precisions; 32 threads is slower and noisier, mainly in the eigendecomposition.
- Reads are faster than on November 22 (GPU 2.0 s against 5.5 s; CPU about 0.9 s against 3.7 s), consistent with bed-reader 1.3.0b3's chunked reads.
- Accuracy against CPU float64 (16 threads): GPU float32 max absolute log10(P) difference 3.42e-3; CPU float32 1.71e-3; CPU float64 at 32 threads 4.4e-10. All configurations rank the top 10 SNPs identically.

## GPU memory sizing (November 25, 2026)

GPU float32 `single_snp` with `leave_out_one_chrom=False`, on an RTX 3060 (12 GB) under WSL2, with the same code as the release-candidate result above. Data: `~/.cache/synthetic-bed-sizing/synth_50000x30000.bed` (50,000 individuals x 30,000 SNPs, made with the same generator), sliced to the first N individuals and the first K SNPs for the kernel (all 30,000 SNPs tested), plus the 14,925 x 100,000 benchmark file. Peak memory is CuPy's memory-pool high-water mark; each run used a fresh process. Script and raw results: `~/deldir/gpu-sizing/` (`measure.py`, `run.sh`, `results.jsonl`).

| Individuals | Kernel SNPs | Rank | CuPy peak (MiB) | Bytes per individual x min(individuals, kernel SNPs) | Seconds |
| ---: | ---: | --- | ---: | ---: | ---: |
| 14,925 | 100,000 | full | 5,950 | 28.0 | 32.0 |
| 16,000 | 30,000 | full | 6,837 | 28.0 | 20.1 |
| 18,000 | 30,000 | full | 8,653 | 28.0 | 26.6 |
| 20,000 | 30,000 | full | 10,683 | 28.0 | 41.8 |
| 22,000 | 30,000 | full | 12,926 | 28.0 | 139.4 (spilled) |
| 14,925 | 5,000 | low | 1,811 | 25.4 | 15.8 |
| 50,000 | 5,000 | low | 6,113 | 25.6 | 18.7 |
| 50,000 | 10,000 | low | 12,988 | 27.2 | 85.6 (spilled) |
| 30,000 | 15,000 | low | 12,039 | 28.1 | 245.1 (spilled) |
| 50,000 | 15,000 | low | 20,626 | 28.8 | 1,540.4 (spilled) |

Findings:

- Peak GPU memory is about 28 bytes x individuals x min(individuals, kernel SNPs), plus about 1 GB of CUDA context and library overhead (`nvidia-smi` minus the pool and the desktop's baseline). Full rank is 28.0 at every size.
- A dedicated 12 GB card handles about 20,000 individuals at full rank, or individuals x kernel SNPs up to about 400 million at low rank (for example, 50,000 individuals with 8,000 kernel SNPs). A display sharing the card (about 2.7 GB here) lowers this.
- Beyond the card's memory, Windows/WSL2 lets CUDA borrow system RAM (WDDM paging): runs still finish but slow down sharply (50,000 x 15,000 took 1,540 s) and push the desktop's GPU memory out. Native Linux would raise an out-of-memory error instead (not tested here).

## Earlier result (November 22, 2026)

The synthetic dataset is `/home/carlk/.cache/synthetic-bed/synth_14925x100000.bed` (14,925 individuals x 100,000 SNPs), with its matching `.pheno.txt`. The generator is `/mnt/c/deldir/synthetic-bed/make_synthetic_bed.py`. `single_snp` used one kernel built from all 100,000 SNPs, which were also the test SNPs.

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

