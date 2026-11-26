"""Draw the GPU benchmark figure: CPU float64 and GPU float32 Manhattan plots.

Reads the per-SNP results (``*.pkl``) and timings (``summary.json``) that
``driver_synthetic.py`` wrote, saves ``doc/source/images/benchmark_manhattan.png``,
and prints the top-5 table and agreement numbers used in
``doc/source/benchmark.rst``. Run it with the interpreter that ran the benchmark,
because the pickles need a compatible pandas.
"""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = Path.home() / ".cache" / "synthetic-bed" / "benchmark-results" / "2026-11-25-dev-checkouts"
DEFAULT_OUTPUT = ROOT / "doc" / "source" / "images" / "benchmark_manhattan.png"
GENOME_WIDE = 5e-8
TOP_COUNT = 5
TEXT = "#191919"  # near black
TEXT_SECONDARY = "#666666"  # mid gray
GRID = "#e5e5e5"  # light gray
THRESHOLD = "#ec4647"  # red, the reference palette's slot 8
# Each panel alternates chromosomes between two steps of one hue.
CPU_COLORS = ("#5a5a5a", "#a8a8a8")  # dark gray, light gray
GPU_COLORS = ("#257de0", "#8dbcf0")  # blue (reference slot 1), light blue


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    summary = json.loads((arguments.results / "summary.json").read_text())
    cpu_seconds = median_total(summary, "cpu-float64-16")
    gpu_seconds = median_total(summary, "gpu-float32-32")
    cpu = pd.read_pickle(arguments.results / "large-14925-cpu-float64-threads16-repeat1.pkl")
    gpu = pd.read_pickle(arguments.results / "large-14925-gpu-float32-threads32-repeat1.pkl")
    assert len(cpu) == len(gpu) == 100_000
    assert set(cpu.SNP) == set(gpu.SNP)
    assert (cpu.PValue > 0).all() and (gpu.PValue > 0).all(), "p = 0 would need special handling"

    both = cpu.set_index("SNP")[["Chr", "ChrPos", "PValue"]].join(gpu.set_index("SNP").PValue.rename("GpuPValue"))
    cpu_log = -np.log10(both.PValue.to_numpy())
    gpu_log = -np.log10(both.GpuPValue.to_numpy())
    correlation = float(np.corrcoef(cpu_log, gpu_log)[0, 1])
    largest_difference = float(np.max(np.abs(cpu_log - gpu_log)))
    top100_overlap = len(set(cpu.nsmallest(100, "PValue").SNP) & set(gpu.nsmallest(100, "PValue").SNP))
    cpu_top = cpu.nsmallest(TOP_COUNT, "PValue").SNP.tolist()
    gpu_top = gpu.nsmallest(TOP_COUNT, "PValue").SNP.tolist()
    cpu_significant = set(cpu.SNP[cpu.PValue < GENOME_WIDE])
    gpu_significant = set(gpu.SNP[gpu.PValue < GENOME_WIDE])

    # The figure says "the same hits" and shares one ranked table between both panels.
    assert cpu_significant == gpu_significant, "the CPU and GPU genome-wide hits differ; reword the figure"
    assert cpu_top == gpu_top, "the CPU and GPU top SNPs differ in order; the shared table would mislead"
    figure = draw(both, cpu_seconds, gpu_seconds, cpu_top, gpu_top, largest_difference, len(cpu_significant))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=150, facecolor="white")
    print(f"wrote {arguments.output}")

    print(f"CPU float64 {cpu_seconds:.1f} s, GPU float32 {gpu_seconds:.1f} s, {cpu_seconds / gpu_seconds:.1f}x")
    print(f"correlation of -log10 p: {correlation:.7f}; largest |difference|: {largest_difference:.2e}")
    print(f"top-100 overlap: {top100_overlap}/100; same top {TOP_COUNT} in order: {cpu_top == gpu_top}")
    print(f"genome-wide significant: CPU {len(cpu_significant)}, GPU {len(gpu_significant)}, same set: {cpu_significant == gpu_significant}")
    print()
    print("Top-5 table (reStructuredText rows):")
    for rank, snp in enumerate(cpu_top, start=1):
        row = both.loc[snp]
        print(f"   * - {rank}\n     - {snp}\n     - {int(row.Chr)}\n     - {row.PValue:.3e}\n     - {row.GpuPValue:.3e}")


def median_total(summary, configuration):
    return statistics.median(summary["timing"][configuration]["stages"]["total"])


def draw(both, cpu_seconds, gpu_seconds, cpu_top, gpu_top, largest_difference, shared_hits):
    chromosomes = sorted(both.Chr.unique())
    # Lay chromosomes end to end, in order, with a small gap between them.
    offsets, ticks, start = {}, [], 0.0
    gap = 0.02 * both.ChrPos.max()
    for chromosome in chromosomes:
        length = both.ChrPos[both.Chr == chromosome].max()
        offsets[chromosome] = start
        ticks.append(start + length / 2)
        start += length + gap
    position = both.Chr.map(offsets) + both.ChrPos
    parity = both.Chr.map({chromosome: index % 2 for index, chromosome in enumerate(chromosomes)})

    cpu_log = -np.log10(both.PValue)
    gpu_log = -np.log10(both.GpuPValue)
    top = max(cpu_log.max(), gpu_log.max()) * 1.12

    # Sized to stay legible when a web page scales the image to about 800 pixels wide.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": TEXT,
                         "axes.labelcolor": TEXT_SECONDARY, "xtick.color": TEXT_SECONDARY,
                         "ytick.color": TEXT_SECONDARY})
    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 2, left=0.065, right=0.985, top=0.79, bottom=0.07,
                               height_ratios=[1.55, 1], hspace=0.34, wspace=0.12)
    axes = [figure.add_subplot(grid[0, 0])]
    axes.append(figure.add_subplot(grid[0, 1], sharey=axes[0]))
    panels = [
        (axes[0], cpu_log, CPU_COLORS, "CPU float64", format_seconds(cpu_seconds), cpu_top),
        (axes[1], gpu_log, GPU_COLORS, "GPU float32 (CuPy)", format_seconds(gpu_seconds), gpu_top),
    ]
    for axis, values, colors, name, duration, top_snps in panels:
        for parity_value, color in enumerate(colors):
            mask = (parity == parity_value).to_numpy()
            axis.scatter(position[mask], values[mask], s=9, color=color, linewidths=0, rasterized=True)
        axis.axhline(-np.log10(GENOME_WIDE), color=THRESHOLD, linewidth=1.2, linestyle=(0, (4, 3)))
        # Number the top SNPs to match the table's ranks.
        for rank, snp in enumerate(top_snps, start=1):
            axis.annotate(str(rank), (position[snp], values[snp]), xytext=(7, 0), textcoords="offset points",
                          fontsize=13, fontweight="bold", color=TEXT, va="center")
        axis.set_ylim(0, top)
        axis.set_xlim(position.min() - gap, position.max() + gap)
        axis.set_xticks([tick for index, tick in enumerate(ticks) if index % 2 == 0],
                        [str(int(chromosome)) for chromosome in chromosomes[::2]], fontsize=12)
        axis.tick_params(axis="x", length=0)
        axis.tick_params(axis="y", labelsize=13)
        axis.set_xlabel("Chromosome", fontsize=13)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.set_title(f"{name}:  {duration}", loc="left", fontsize=21, fontweight="bold", color=TEXT, pad=10)
    axes[0].set_ylabel("−log10(p)", fontsize=14)
    axes[1].tick_params(axis="y", labelleft=False)

    speedup = cpu_seconds / gpu_seconds
    figure.text(0.065, 0.925, f"Same peaks, {speedup:.0f}× faster on the GPU", fontsize=32, fontweight="bold", color=TEXT)
    figure.text(0.065, 0.865, f"FaST-LMM single_snp on 14,925 individuals × 100,000 SNPs. "
                f"Both find the same {shared_hits} genome-wide hits.", fontsize=16, color=TEXT_SECONDARY)

    table_axis = figure.add_subplot(grid[1, :])
    table_axis.axis("off")
    columns = [(0.0, "Rank", "left"), (0.08, "SNP", "left"), (0.22, "Chromosome", "left"),
               (0.52, "p, CPU float64", "right"), (0.72, "p, GPU float32", "right")]
    row_height = 1 / (len(cpu_top) + 1.6)
    header_y = 1 - row_height / 2
    for x, label, alignment in columns:
        table_axis.text(x, header_y, label, ha=alignment, va="center", fontsize=15, fontweight="bold", color=TEXT_SECONDARY)
    table_axis.axhline(1 - row_height, xmin=0, xmax=0.73, color=GRID, linewidth=1.2)
    for row_index, snp in enumerate(cpu_top):
        y = header_y - (row_index + 1.1) * row_height
        row = both.loc[snp]
        cells = [str(row_index + 1), snp, str(int(row.Chr)), scientific(row.PValue), scientific(row.GpuPValue)]
        for (x, _, alignment), cell in zip(columns, cells):
            table_axis.text(x, y, cell, ha=alignment, va="center", fontsize=16, color=TEXT)
    table_axis.set_xlim(0, 1)
    table_axis.set_ylim(0, 1)

    figure.text(0.065, 0.015, "Synthetic data. NVIDIA RTX 3060 (12 GB) vs. AMD Ryzen 9 7950X (16 threads); "
                f"median of three runs; −log10(p) differs by at most {largest_difference:.3f}. fastlmm.github.io",
                fontsize=11.5, color=TEXT_SECONDARY)
    return figure


def scientific(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"${mantissa} \\times 10^{{{int(exponent)}}}$"


def format_seconds(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes} min {remainder} s"


if __name__ == "__main__":
    main()
