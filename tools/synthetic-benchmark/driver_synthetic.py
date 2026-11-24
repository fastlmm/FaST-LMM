"""Run the CPU float32 eigendecomposition benchmark configurations."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
DEFAULT_OUTPUT = (
    Path.home()
    / ".cache"
    / "synthetic-bed"
    / "benchmark-results"
    / "2026-11-22-cpu-f32-eigh"
)
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_BED = Path.home() / ".cache" / "synthetic-bed" / "synth_14925x100000.bed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individuals", type=int, default=14925)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--bed", type=Path, default=DEFAULT_BED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_manifest(arguments)

    run_configuration(arguments, "gpu", "float32", 32, arguments.repeats)
    cpu_results = {
        threads: run_configuration(
            arguments, "cpu", "float32", threads, arguments.repeats
        )
        for threads in (8, 16, 32)
    }
    fastest_threads = min(
        cpu_results,
        key=lambda threads: statistics.median(
            result["timed"]["seconds"]["total"] for result in cpu_results[threads]
        ),
    )
    print(f"Fastest CPU float32 setting: {fastest_threads} threads", flush=True)
    run_configuration(
        arguments, "cpu", "float64", fastest_threads, arguments.repeats
    )
    if fastest_threads != 32:
        run_configuration(arguments, "cpu", "float64", 32, 1)


def write_manifest(arguments):
    repositories = {
        "fastlmm": ROOT,
        "pysnptools": Path("/home/carlk/programs/PySnpTools"),
        "bed-reader": Path("/home/carlk/programs/bed-reader"),
    }
    manifest = {
        "python": str(arguments.python.resolve()),
        "bed": str(arguments.bed.resolve()),
        "phenotype": str(arguments.bed.with_suffix(".pheno.txt").resolve()),
        "repositories": {
            name: {
                "path": str(path),
                "branch": git(path, "branch", "--show-current"),
                "commit": git(path, "rev-parse", "HEAD"),
            }
            for name, path in repositories.items()
        },
    }
    destination = arguments.output / "manifest.json"
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError(f"{destination} describes a different run. Use a new --output directory.")
        return
    if any(arguments.output.glob("large-*.json")):
        raise ValueError("Benchmark results have no manifest. Use a new --output directory.")
    destination.write_text(json.dumps(manifest, indent=2))


def git(repository, *arguments):
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def run_configuration(arguments, backend, precision, threads, repeat_count):
    run_worker(arguments, backend, precision, threads, "warmup", "warmup")
    results = []
    for repeat in range(1, repeat_count + 1):
        repeat = f"repeat{repeat}"
        run_worker(arguments, backend, precision, threads, "timed", repeat)
        destination = result_path(
            arguments, backend, precision, threads, repeat, ".json"
        )
        results.append(json.loads(destination.read_text()))
    return results


def result_path(arguments, backend, precision, threads, repeat, suffix):
    name = (
        f"large-{arguments.individuals}-{backend}-{precision}"
        f"-threads{threads}-{repeat}"
    )
    return arguments.output / f"{name}{suffix}"


def run_worker(arguments, backend, precision, threads, label, repeat):
    destination = result_path(
        arguments, backend, precision, threads, repeat, ".json"
    )
    name = destination.stem
    if destination.exists():
        existing = json.loads(destination.read_text())
        assert existing["arguments"]["bed"] == str(arguments.bed)
        seconds = existing[label]["seconds"]["total"]
        print(f"Reuse {name}: {seconds:.3f} s", flush=True)
        return

    if backend == "gpu":
        wait_for_gpu(arguments.output, name)
    print(f"Start {name}", flush=True)
    command = [
        str(arguments.python),
        "-u",
        str(SCRIPT / "benchmark.py"),
        "--size",
        "large",
        "--individuals",
        str(arguments.individuals),
        "--markers",
        "100000",
        "--bed",
        str(arguments.bed),
        "--precision",
        precision,
        "--backend",
        backend,
        "--threads",
        str(threads),
        "--label",
        label,
        "--repeat",
        repeat,
        "--output",
        str(arguments.output),
    ]
    log = result_path(arguments, backend, precision, threads, repeat, ".log")
    with log.open("w") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT
        )
        start = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - start > 1800:
                process.terminate()
                process.wait(timeout=30)
                raise TimeoutError(f"{name} exceeded the 30-minute worker limit")
            if backend == "gpu":
                record_gpu(arguments.output, name, "DURING")
            time.sleep(10)
        assert process.returncode == 0, f"{name} failed; inspect {log}"
    result = json.loads(destination.read_text())
    seconds = result[label]["seconds"]
    print(f"Completed {name}: {seconds['total']:.3f} s; {seconds}", flush=True)


def wait_for_gpu(output, name):
    for _ in range(24):
        status = record_gpu(output, name, "BEFORE")
        utilization = int(status.rsplit(",", 1)[1].strip().split()[0])
        if utilization <= 25:
            return
        print(f"GPU busy: {status}", flush=True)
        time.sleep(5)
    raise RuntimeError("Shared GPU busy for two minutes; resume later.")


def record_gpu(output, name, phase):
    status = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    with (output / "gpu-status.log").open("a") as stream:
        stream.write(f"{phase} {name} {status}\n")
    return status


if __name__ == "__main__":
    main()
