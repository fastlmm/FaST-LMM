#!/usr/bin/env bash
# Execute FaST-LMM notebooks per tests/notebooks.toml, from copies (the repo is untouched).
#   full       -> run every cell
#   prefix     -> run cells before the markdown heading `stop_before_heading`
#   historical -> skip and report as excluded
# The scaling notebook's hard-coded c:\deldir paths are replaced with a portable
# temp-folder path in the copy (the same edit proposed for the notebook itself).
# Usage: bash fastlmm-notebooks.sh [name ...]   e.g. bash fastlmm-notebooks.sh fastlmm2021
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$HOME/programs/FaST-LMM}
ENV_DIR=${ENV_DIR:-$HOME/.cache/fastlmm-release-2026/fl-notebook314}
RUN_DIR=${RUN_DIR:-$HOME/.cache/fastlmm-release-2026/nbrun-fastlmm}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

(cd "$REPO" && UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync -q --python 3.14 --frozen --all-extras --group notebook) || exit 1
mkdir -p "$RUN_DIR"

"$ENV_DIR/bin/python" - "$REPO" "$RUN_DIR" "$@" <<'PY'
import json, sys, subprocess, time, tomllib
from pathlib import Path
repo, run = Path(sys.argv[1]), Path(sys.argv[2]); only = set(sys.argv[3:])
policy = tomllib.loads((repo / "tests/notebooks.toml").read_text())["notebook"]
unknown = only - {Path(entry["path"]).stem for entry in policy}
if unknown:
    sys.exit(f"Unknown notebook name(s): {', '.join(sorted(unknown))}")
py = Path(sys.executable)

def portable(nb):
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        s = s.replace("cache_top = r'c:\\deldir'",
                      "import os, tempfile\ncache_top = os.path.join(tempfile.gettempdir(), 'fastlmm_scale')")
        # Match the indented code line only; the same text also appears in a comment.
        s = s.replace("     return ip_pid, r'c:/deldir/peertopeer1/{0}'.format(ip_pid)",
                      "     import os, tempfile\n     return ip_pid, os.path.join(tempfile.gettempdir(), 'fastlmm_scale', 'peertopeer1', '{0}'.format(ip_pid))")
        s = s.replace("#Change the line 'return ip_pid, r'c:/deldir/peertopeer1/{0}'.format(ip_pid)'",
                      "#Change the path in id_and_path_function (and cache_top above)")
        c["source"] = s.splitlines(keepends=True)
    return nb

failed = False
for entry in policy:
    src = repo / entry["path"]; name = src.stem
    if only and name not in only:
        continue
    if entry["mode"] == "historical":
        print(f"{name}: EXCLUDED (historical) - {entry['reason']}", flush=True); continue
    nb = json.loads(src.read_text(encoding="utf-8"))
    note = "all cells"
    if entry["mode"] == "prefix":
        stop = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"
                    and "".join(c["source"]).lstrip().startswith(entry["stop_before_heading"]))
        omitted = sum(c["cell_type"] == "code" for c in nb["cells"][stop:])
        nb["cells"] = nb["cells"][:stop]
        note = f"stopped before '{entry['stop_before_heading']}', {omitted} code cells omitted"
    nb = portable(nb)
    inp = run / f"{name}.ipynb"; inp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t = time.time()
    r = subprocess.run([str(py.parent / "jupyter"), "nbconvert", "--to", "notebook", "--execute",
                        "--ExecutePreprocessor.timeout=-1", "--output", f"{name}.executed.ipynb", inp.name],
                       cwd=run, capture_output=True, text=True)
    (run / f"{name}.log").write_text(r.stdout + r.stderr)
    failed = failed or r.returncode != 0
    status = "ok" if r.returncode == 0 else f"FAILED (see {run}/{name}.log)"
    print(f"{name}: {status} [{note}] {time.time() - t:.0f}s", flush=True)
sys.exit(1 if failed else 0)
PY
