"""Generate a non-identifying synthetic PLINK BED file for timing FaST-LMM.

The defaults make the 14,925 individuals x 100,000 SNPs file used by FaST-LMM's
GPU benchmark (doc/source/benchmark.rst). The genotype model follows
PySnpTools' ``snp_gen`` (Balding-Nichols population structure with uniform
ancestral allele frequencies), but is written SNP by SNP with bed-reader's
``create_bed`` so memory stays small, and uses a local ``default_rng``.

Unlike PySnpTools' ``SnpGen`` (21.8% missing; ~98.5% of SNPs with MAF < 0.01),
this resembles a QC-filtered GWAS array: common SNPs and little missing data.

Outputs ``<stem>.bed/.bim/.fam``, ``<stem>.pheno.txt`` (FID IID value), and
``<stem>.provenance.json`` in the first output folder, then copies them to the
other output folders and checks the SHA-256 digests match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from bed_reader import create_bed

DEFAULT_OUTPUTS = [Path.home() / ".cache" / "synthetic-bed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iid-count", type=int, default=14_925)
    parser.add_argument("--sid-count", type=int, default=100_000)
    parser.add_argument("--chrom-count", type=int, default=22)
    parser.add_argument("--fst", type=float, default=0.1)
    parser.add_argument("--maf-low", type=float, default=0.05)
    parser.add_argument("--maf-high", type=float, default=0.5)
    parser.add_argument("--missing-rate", type=float, default=0.01)
    parser.add_argument("--causal-count", type=int, default=1_000, help="SNPs in the polygenic background")
    parser.add_argument("--h2", type=float, default=0.4, help="Variance explained by the polygenic background")
    parser.add_argument("--large-count", type=int, default=20, help="Large-effect SNPs, disjoint from the background")
    parser.add_argument("--large-h2", type=float, default=0.1, help="Variance explained by the large-effect SNPs together")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", type=Path, action="append", help="Output folder (repeatable). Default: ~/.cache/synthetic-bed")
    arguments = parser.parse_args()
    outputs = arguments.output or DEFAULT_OUTPUTS

    iid_count, sid_count, chrom_count = arguments.iid_count, arguments.sid_count, arguments.chrom_count
    assert iid_count > 0 and sid_count > 0
    assert 1 <= chrom_count <= sid_count
    assert 0.0 < arguments.fst < 1.0
    assert 0.0 < arguments.maf_low <= arguments.maf_high <= 0.5
    assert 0.0 <= arguments.missing_rate < 1.0
    assert 0 < arguments.causal_count and 0 < arguments.large_count
    assert arguments.causal_count + arguments.large_count <= sid_count
    assert 0.0 <= arguments.h2 and 0.0 <= arguments.large_h2 and arguments.h2 + arguments.large_h2 <= 1.0

    stem = f"synth_{iid_count}x{sid_count}"
    primary = outputs[0]
    primary.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(arguments.seed)

    # Individuals: each is randomly assigned to one of two populations.
    population = (rng.random(iid_count) < 0.5).astype(np.int8)
    fid = [f"pop{population_index}" for population_index in population]
    iid = [f"i_{iid_index}" for iid_index in range(iid_count)]

    # SNPs: spread evenly over chromosomes, 1,000 bp apart.
    sid_per_chrom = -(-sid_count // chrom_count)  # ceiling division
    sid_index = np.arange(sid_count)
    chromosome = (1 + sid_index // sid_per_chrom).astype(str)
    bp_position = 1_000 * (1 + sid_index % sid_per_chrom)
    sid = [f"snp_{index}" for index in sid_index]

    chosen = rng.choice(sid_count, arguments.causal_count + arguments.large_count, replace=False)
    large = np.sort(chosen[: arguments.large_count])
    causal = np.sort(chosen[arguments.large_count :])
    role = np.zeros(sid_count, dtype=np.int8)  # 0: null, 1: polygenic background, 2: large effect
    role[causal] = 1
    role[large] = 2
    genetic_value = {1: np.zeros(iid_count), 2: np.zeros(iid_count)}

    properties = {"fid": fid, "iid": iid, "sid": sid, "chromosome": chromosome, "bp_position": bp_position,
                  "allele_1": ["A"] * sid_count, "allele_2": ["G"] * sid_count}
    bed_path = primary / f"{stem}.bed"
    in_pop1 = population == 1
    start = time.perf_counter()
    # count_A1=False matches how benchmark.py reads the file, so values read back exactly as generated.
    with create_bed(bed_path, iid_count=iid_count, sid_count=sid_count, properties=properties, count_A1=False, major="SNP") as bed_writer:
        for index in range(sid_count):
            ancestral = rng.uniform(arguments.maf_low, arguments.maf_high)
            # Balding-Nichols: each population's frequency is Beta-distributed around the ancestral one.
            alpha = ancestral * (1.0 - arguments.fst) / arguments.fst
            beta = (1.0 - ancestral) * (1.0 - arguments.fst) / arguments.fst
            frequency = rng.beta(alpha, beta, size=2)
            genotype = rng.binomial(2, np.where(in_pop1, frequency[1], frequency[0])).astype(np.int8)
            if role[index] != 0:
                standardized = genotype.astype(np.float64)
                standard_deviation = standardized.std()
                assert standard_deviation > 0, f"causal SNP {index} is monomorphic; try another seed"
                genetic_value[role[index]] += rng.normal() * (standardized - standardized.mean()) / standard_deviation
            genotype[rng.random(iid_count) < arguments.missing_rate] = -127  # bed-reader's int8 missing value
            bed_writer.write(genotype)
            if (index + 1) % 10_000 == 0:
                print(f"{index + 1:,}/{sid_count:,} SNPs, {time.perf_counter() - start:.1f} s", flush=True)
    generate_seconds = time.perf_counter() - start

    # Phenotype: each genetic component scaled to its variance, plus noise with the remaining variance.
    background, large_value = ((value - value.mean()) / value.std() for value in (genetic_value[1], genetic_value[2]))
    noise_variance = 1.0 - arguments.h2 - arguments.large_h2
    phenotype = np.sqrt(arguments.h2) * background + np.sqrt(arguments.large_h2) * large_value + np.sqrt(noise_variance) * rng.normal(size=iid_count)
    pheno_path = primary / f"{stem}.pheno.txt"
    with pheno_path.open("w", newline="\n") as stream:
        for fid_item, iid_item, value in zip(fid, iid, phenotype):
            stream.write(f"{fid_item}\t{iid_item}\t{value:.10g}\n")

    files = [primary / f"{stem}.{suffix}" for suffix in ["bed", "bim", "fam", "pheno.txt"]]
    digests = {path.name: sha256(path) for path in files}
    provenance = {"generator": str(Path(__file__).resolve()), "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items() if key != "output"},
                  "count_A1": False, "population_1_count": int(in_pop1.sum()), "large_sid_index": large.tolist(), "causal_sid_index": causal.tolist(),
                  "generate_seconds": generate_seconds, "sha256": digests}
    provenance_path = primary / f"{stem}.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    files.append(provenance_path)
    digests[provenance_path.name] = sha256(provenance_path)

    for other in outputs[1:]:
        other.mkdir(parents=True, exist_ok=True)
        for path in files:
            copy = other / path.name
            shutil.copyfile(path, copy)
            copied_digest = sha256(copy)
            assert copied_digest == digests[path.name], f"digest mismatch for {copy}"
    for output in outputs:
        print(f"wrote {output}")
    print(json.dumps(digests, indent=2))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
