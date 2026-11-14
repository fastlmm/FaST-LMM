"""Copy executed outputs (and SingleSnpScale's portable source edits) into the repo notebooks.

Usage: python refresh-notebook-outputs.py REPO RUN_DIR
Keeps each notebook's metadata and line endings. In SingleSnpScale, cells after the
executed prefix keep their committed outputs, and verbose file-I/O stdout logging
(which records local IPs and temp paths) is dropped, keeping the result tables.
"""
import json
import sys
from pathlib import Path
repo, run = Path(sys.argv[1]), Path(sys.argv[2])
for name in ["FaST-LMM", "fastlmm2021", "heritability_si", "SingleSnpScale"]:
    path = repo / "doc/ipynb" / f"{name}.ipynb"
    raw = path.read_bytes()
    crlf = b"\r\n" in raw
    nb = json.loads(raw.decode("utf-8"))
    ex = json.loads((run / f"{name}.executed.ipynb").read_text(encoding="utf-8"))["cells"]
    for i, cb in enumerate(ex):
        ca = nb["cells"][i]
        assert ca["cell_type"] == cb["cell_type"], (name, i)
        if ca["cell_type"] != "code":
            continue
        ca["source"] = cb["source"]
        outs = cb["outputs"]
        if name == "SingleSnpScale" and any(
                o["output_type"] == "stream" and ("open_read(" in "".join(o["text"]) or "open_write(" in "".join(o["text"]))
                for o in outs):
            outs = [o for o in outs if o["output_type"] != "stream"]
        ca["outputs"] = outs
        ca["execution_count"] = cb["execution_count"]
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    for bad in ("/tmp/", "10.0.0.", "/home/", "deldir"):
        if bad in json.dumps([c.get("outputs", []) for c in nb["cells"]]):
            print(f"  warning: {name} outputs contain {bad!r}")
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))
    print(name, "updated", "crlf" if crlf else "lf")
