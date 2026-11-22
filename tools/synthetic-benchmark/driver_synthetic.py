"""Run sequential CPU/GPU and float32/float64 single_snp comparisons."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUTPUT = Path.home() / ".cache" / "synthetic-bed" / "benchmark-results"
OUTPUT.mkdir(parents=True, exist_ok=True)
PYTHON = "/home/carlk/.cache/gpu-read-share/venv/bin/python"
BED = "/home/carlk/.cache/synthetic-bed/synth_14925x100000.bed"
parser = argparse.ArgumentParser()
parser.add_argument("--individuals", type=int, default=14925)
parser.add_argument("--precisions", nargs="+", choices=["float32", "float64"], default=["float32", "float64"])
parser.add_argument("--backends", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
parser.add_argument("--repeats", type=int, default=3)
arguments = parser.parse_args()
os.environ["PYTHONPATH"] = f"{ROOT}:/home/carlk/programs/PySnpTools"
for precision in arguments.precisions:
    for repeat in range(1, arguments.repeats + 1):
        for backend in arguments.backends if repeat % 2 else list(reversed(arguments.backends)):
            if backend == "gpu" and precision == "float64":
                continue
            identifier = f"synthetic-{repeat}"
            name = f"large-{arguments.individuals}-{backend}-{precision}-{identifier}"
            destination = OUTPUT / f"{name}.json"
            if destination.exists():
                existing = json.loads(destination.read_text())
                assert existing["arguments"]["bed"] == BED
                print(f"Reuse {name}: {existing['timed']['seconds']['total']:.3f} s", flush=True)
                continue
            status = "CPU backend; no GPU required"
            if backend == "gpu":
                for attempt in range(24):
                    status = subprocess.check_output(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
                    with (OUTPUT / "large-gpu-status.log").open("a") as stream:
                        stream.write(f"BEFORE {name} {status}\n")
                    utilization = int(status.rsplit(",", 1)[1].strip().split()[0])
                    if utilization <= 25:
                        break
                    print(f"GPU busy: {status}", flush=True)
                    time.sleep(5)
                else:
                    raise RuntimeError("Shared GPU busy for two minutes; resume later.")
            print(f"Start {name}: {status}", flush=True)
            command = [PYTHON, "-u", str(SCRIPT / "benchmark.py"), "--size", "large", "--individuals", str(arguments.individuals), "--markers", "100000", "--bed", BED, "--precision", precision, "--backend", backend, "--repeat", identifier]
            with (OUTPUT / f"{name}.log").open("w") as output:
                process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                start = time.monotonic()
                seen_warmup = False
                while process.poll() is None:
                    if time.monotonic() - start > 1800:
                        process.terminate()
                        process.wait(timeout=30)
                        raise TimeoutError(f"{name} exceeded the 30-minute worker limit")
                    if backend == "gpu":
                        status = subprocess.check_output(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
                        with (OUTPUT / "large-gpu-status.log").open("a") as stream:
                            stream.write(f"DURING {name} {status}\n")
                    if not seen_warmup:
                        for line in (OUTPUT / f"{name}.log").read_text().splitlines():
                            if line.startswith('{"label": "warmup"'):
                                warmup = json.loads(line)
                                print(f"{name} warm-up: {warmup['seconds']['total']:.3f} s", flush=True)
                                seen_warmup = True
                    time.sleep(10)
                assert process.returncode == 0, f"{name} failed; inspect its log"
            result = json.loads(destination.read_text())
            print(f"Completed {name}: {result['timed']['seconds']['total']:.3f} s; {result['timed']['seconds']}", flush=True)
