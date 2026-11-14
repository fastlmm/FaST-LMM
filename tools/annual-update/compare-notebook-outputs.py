"""Compare text outputs of executed notebook copies with the committed notebooks.

Usage: python compare-notebook-outputs.py REPO RUN_DIR NAME [NAME ...]
Ignores timings, memory addresses, blank-line splits, and warnings; prints each
code cell whose remaining output changed, with the changed lines.
"""
import difflib
import json
import re
import sys
from pathlib import Path


def text(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream" and o.get("name") == "stdout":
            parts.append("".join(o["text"]))
        elif o["output_type"] in ("execute_result", "display_data"):
            d = o.get("data", {})
            parts.append("".join(d["text/plain"]) if "text/plain" in d else "[non-text]")
        elif o["output_type"] == "error":
            parts.append(f"ERROR {o['ename']}: {o['evalue']}")
    s = "\n".join(parts)
    s = re.sub(r"0x[0-9a-fA-F]+", "0x?", s)
    s = re.sub(r"(CPU times|Wall time):.*", r"\1: T", s)
    s = re.sub(r"\d+(\.\d+)?\s?(seconds|sec|ms|µs|s)\b", "T", s)
    return [line.rstrip() for line in s.splitlines() if line.strip()]


def main():
    repo, run = Path(sys.argv[1]), Path(sys.argv[2])
    for name in sys.argv[3:]:
        committed = next(repo.rglob(f"{name}.ipynb"))
        a = json.loads(committed.read_text(encoding="utf-8"))["cells"]
        b = json.loads((run / f"{name}.executed.ipynb").read_text(encoding="utf-8"))["cells"]
        changed = 0
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca["cell_type"] != "code":
                continue
            ta, tb = text(ca), text(cb)
            if ta != tb:
                changed += 1
                print(f"--- {name} cell {i}")
                for line in list(difflib.unified_diff(ta, tb, lineterm="", n=0))[2:14]:
                    print("   ", line[:160])
        print(f"===== {name}: {changed} code cells with changed output\n")


if __name__ == "__main__":
    main()
