"""Reproducible, synthetic end-to-end single_snp benchmark."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path.home() / ".cache" / "synthetic-bed" / "benchmark-results"
LARGE = Path("/home/carlk/.cache/synthetic-bed/synth_14925x100000.bed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["mouse", "large"], required=True)
    parser.add_argument("--precision", choices=["float32", "float64"], required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--individuals", type=int, default=15000)
    parser.add_argument("--markers", type=int, default=100000)
    parser.add_argument("--repeat", default="pilot")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--label", choices=["warmup", "timed"], default="timed")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bed", type=Path, default=LARGE)
    parser.add_argument("--pheno", type=Path)
    arguments = parser.parse_args()
    os.environ["ARRAY_MODULE"] = "numpy" if arguments.backend == "cpu" else "cupy"
    os.environ["ARRAY_DTYPE"] = arguments.precision
    thread_count = arguments.threads
    for variable in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "PST_NUM_THREADS", "NUM_THREADS"]:
        os.environ[variable] = str(thread_count)
    import numpy as np

    cp = None
    if arguments.backend == "gpu":
        import cupy as cp
    import scipy.linalg
    import fastlmm
    import pysnptools
    import bed_reader
    from bed_reader import open_bed
    from fastlmm.association import single_snp
    from fastlmm.util import example_file
    from pysnptools.snpreader import Bed
    from pysnptools.snpreader import Pheno
    from pysnptools.kernelreader import SnpKernel
    from pysnptools.standardizer import Standardizer
    import pysnptools.util._array as array_utils
    from threadpoolctl import threadpool_info

    assert str(ROOT) in fastlmm.__file__
    assert "/home/carlk/programs/PySnpTools/" in pysnptools.__file__
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.size == "mouse":
        bed = Bed(example_file("tests/datasets/mouse/alldata.*", "*.bed"), count_A1=False)
        phenotype_path = example_file("tests/datasets/mouse/pheno_10_causals.txt")
        phenotype = phenotype_path
        assert (bed.iid_count, bed.sid_count) == (1940, 10150)
    else:
        bed = Bed(str(arguments.bed), count_A1=False)
        assert bed.iid_count == arguments.individuals
        assert bed.sid_count >= arguments.markers
        if bed.sid_count > arguments.markers:
            bed = bed[:, : arguments.markers]
        phenotype_path = arguments.pheno or arguments.bed.with_suffix(".pheno.txt")
        phenotype = Pheno(str(phenotype_path)).read()
        assert np.array_equal(phenotype.iid, bed.iid), "synthetic phenotype IIDs must match BED order"
    dimensions = {"individuals": bed.iid_count, "snps": bed.sid_count}
    module = importlib.import_module("fastlmm.association.single_snp")
    durations = defaultdict(float)
    calls = defaultdict(int)
    stack = []
    read_array_depth = 0
    evidence = []
    xp = cp if arguments.backend == "gpu" else np

    def synchronize():
        if cp is not None:
            cp.cuda.runtime.deviceSynchronize()

    @contextmanager
    def stage(name):
        synchronize()
        entry = {"name": name, "children": 0.0}
        start = time.perf_counter()
        stack.append(entry)
        try:
            yield
        finally:
            synchronize()
            elapsed = time.perf_counter() - start
            stack.pop()
            durations[name] += elapsed - entry["children"]
            calls[name] += 1
            if stack:
                stack[-1]["children"] += elapsed

    def wrap(target, attribute, name, static=False, export=False):
        original_function = getattr(target, attribute)

        def measured(*args, **kwargs):
            with stage(name):
                result = original_function(*args, **kwargs)
                if export:
                    # CUDA array export flushes CubeCL work before device sync.
                    exported = cp.asarray(result)
                    assert exported.dtype == np.dtype(arguments.precision)
                    del exported
                return result

        setattr(target, attribute, staticmethod(measured) if static else measured)

    wrap(Standardizer, "_standardize_unit_and_beta", "standardize", static=True)
    original_read = open_bed.read
    read_accepts_xp = "xp" in inspect.signature(original_read).parameters

    def measured_read(self, *args, **kwargs):
        if read_accepts_xp:
            kwargs["xp"] = np
        with stage("read"):
            return original_read(self, *args, **kwargs)

    open_bed.read = measured_read
    original_read_array = Bed._read_array

    def measured_read_array(self, *args, **kwargs):
        nonlocal read_array_depth
        read_array_depth += 1
        try:
            return original_read_array(self, *args, **kwargs)
        finally:
            read_array_depth -= 1

    Bed._read_array = measured_read_array
    original_asarray = array_utils.asarray

    def measured_asarray(values, dtype, order, xp, copy=False):
        if read_array_depth and xp is cp and isinstance(values, np.ndarray):
            with stage("copy"):
                return original_asarray(values, dtype, order, xp, copy=copy)
        return original_asarray(values, dtype, order, xp, copy=copy)

    array_utils.asarray = measured_asarray
    wrap(SnpKernel, "_read_with_standardizing", "kernel")
    wrap(module, "_snp_tester", "association")
    wrap(module, "_fit_h2", "h2_fit")
    if cp is not None:
        linalg = cp.linalg
    elif arguments.precision == "float32":
        linalg = scipy.linalg
    else:
        linalg = np.linalg
    original_eigh = linalg.eigh

    def measured_eigh(matrix, *args, **kwargs):
        if matrix.shape[0] > 100:
            evidence.append({"shape": list(matrix.shape), "dtype": str(matrix.dtype)})
            with stage("eigendecomposition"):
                return original_eigh(matrix, *args, **kwargs)
        return original_eigh(matrix, *args, **kwargs)

    linalg.eigh = measured_eigh
    # GB_goal's minimum block is iid_count. Both pipelines use that same block.
    options = dict(test_snps=bed, pheno=phenotype, leave_out_one_chrom=False, force_full_rank=True, count_A1=False, GB_goal=0.5, xp=xp, dtype=arguments.precision)

    def run(label):
        durations.clear()
        calls.clear()
        evidence.clear()
        synchronize()
        start = time.perf_counter()
        frame = single_snp(**options)
        synchronize()
        total = time.perf_counter() - start
        stages = dict(durations)
        stages["other"] = total - sum(stages.values())
        stages["total"] = total
        assert len(evidence) == 1, evidence
        assert evidence[0] == {"shape": [bed.iid_count, bed.iid_count], "dtype": arguments.precision}
        assert calls["read"] > 0 and calls["standardize"] > 0
        if arguments.backend == "gpu":
            assert calls["copy"] > 0
        else:
            assert calls["copy"] == 0
        print(json.dumps({"label": label, "seconds": stages, "calls": dict(calls)}), flush=True)
        return frame, {"seconds": stages, "calls": dict(calls), "eigh": list(evidence)}

    frame, measurement = run(arguments.label)
    name = f"{arguments.size}-{arguments.individuals if arguments.size == 'large' else 1940}-{arguments.backend}-{arguments.precision}-threads{thread_count}-{arguments.repeat}"
    if arguments.label == "timed":
        frame.to_pickle(arguments.output / f"{name}.pkl")
    blas = threadpool_info()
    assert blas and all(pool["num_threads"] == thread_count for pool in blas), blas
    packages = ["fastlmm", "pysnptools", "bed-reader", "numpy", "scipy"]
    if cp is not None:
        packages.append("cupy-cuda13x")
    result = dict(
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
        dimensions=dimensions,
        **{arguments.label: measurement},
        versions={package: importlib.metadata.version(package) for package in packages},
        imports={package: loaded.__file__ for package, loaded in [("fastlmm", fastlmm), ("pysnptools", pysnptools), ("bed_reader", bed_reader)]},
        blas=blas,
        gpu_memory=cp.cuda.runtime.memGetInfo() if cp is not None else None,
        settings={
            "backend": arguments.backend,
            "leave_out_one_chrom": False,
            "force_full_rank": True,
            "GB_goal": 0.5,
            "reader_threads": thread_count,
            "blas_threads": thread_count,
            "cpu_count": os.cpu_count(),
            "phenotype": str(phenotype_path),
            "phenotype_iid_order_checked": True,
            "bed_reader_read_backend": "CPU",
            "explicit_numpy_read_used_for_legacy_reader": read_accepts_xp,
        },
    )
    (arguments.output / f"{name}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
