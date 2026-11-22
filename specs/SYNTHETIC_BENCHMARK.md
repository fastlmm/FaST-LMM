# Synthetic FaST-LMM benchmark

## Quick result (November 21, 2026)

The dataset is `/home/carlk/.cache/synthetic-bed/synth_14925x100000.bed`. The benchmark reads the matching `.pheno.txt` and checks individual order. The generator is `/mnt/c/deldir/synthetic-bed/make_synthetic_bed.py`.

The run used FaST-LMM at `c3af42c` and PySnpTools at `d839fe2`. Both backends used the cached bed-reader 1.3.0b1 with CPU reads selected explicitly. The harness is in [tools/synthetic-benchmark](../tools/synthetic-benchmark).

Completed float32 medians across three repeats:

| Backend | Median seconds |
| --- | ---: |
| CPU | 384.850 |
| GPU | 30.277 |

This is a 12.7x GPU speedup for this `single_snp` workload. CPU and GPU float32 ranked the same top 10 SNPs in the same order. CPU float32 and CPU float64 also agreed on that ranking, with a maximum absolute p-value difference of `2.19e-6`.

Only one float64 run per backend completed before the run was stopped: CPU 420.284 s, GPU 594.400 s. The remaining float64 repeats were interrupted, so these are not stable timing estimates. No GPU float64 runs are needed in future benchmark batches.

## Resume an overnight batch

From the FaST-LMM checkout, use the cached environment to run CPU and GPU float32, then CPU float64 as the precision reference.

```bash
/home/carlk/.cache/gpu-read-share/venv/bin/python -u tools/synthetic-benchmark/driver_synthetic.py --repeats 3
```

The driver runs one worker at a time and reuses completed JSON results. With the saved results in the cache, this command only runs CPU float64 repeats 2 and 3. The CPU benchmark uses a 32-thread setting.

Timing JSON, association results, and logs live in `~/.cache/synthetic-bed/benchmark-results/`. A stopped repeat may leave a partial `.log`; it has no completed JSON result and will be run again when resumed.
