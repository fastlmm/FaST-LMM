#!/usr/bin/env bash
# Recreate the clean notebook environment and execute PySnpTools notebooks.
# Usage: bash notebook-env.sh [notebook.ipynb ...]   (default: tutorial paper Dist)
# Outputs go to $ENV_DIR/../nbrun as <name>.executed.ipynb; the repo is untouched.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# Keep the environment on the Linux filesystem; venvs on /mnt/c are very slow.
ENV_DIR=${ENV_DIR:-$HOME/.cache/fastlmm-release-2026/pst-nb}
RUN_DIR=${RUN_DIR:-$HOME/.cache/fastlmm-release-2026/nbrun}
PST=${PST:-$HOME/programs/PySnpTools}
FASTLMM_VERSION=${FASTLMM_VERSION:-1.0.0}
export QCTOOLPATH=${QCTOOLPATH:-$HOME/.local/bin/qctool}

(cd "$PST" && UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync -q --frozen --all-extras --group notebook --python 3.14)
VIRTUAL_ENV="$ENV_DIR" uv pip install -q "fastlmm==$FASTLMM_VERSION"

mkdir -p "$RUN_DIR"
nbs=("$@"); [[ ${#nbs[@]} -eq 0 ]] && nbs=(tutorial paper Dist)
failed=0
for nb in "${nbs[@]}"; do
  nb=${nb%.ipynb}
  cp "$PST/doc/ipynb/$nb.ipynb" "$RUN_DIR/"
  (cd "$RUN_DIR" && "$ENV_DIR/bin/jupyter" nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=1800 --output "$nb.executed.ipynb" "$nb.ipynb" > "$nb.log" 2>&1) \
     && echo "$nb: ok" || { echo "$nb: FAILED (see $RUN_DIR/$nb.log)"; failed=1; }
done
exit "$failed"
