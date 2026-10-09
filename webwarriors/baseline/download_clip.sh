#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_ID="openai/clip-vit-large-patch14-336"
MODEL_DIR="${REPO_ROOT}/clip-vit-large-patch14-336"

echo
echo "Model : ${MODEL_ID}"
echo "Target: ${MODEL_DIR}"
echo

if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "[INFO] huggingface_hub not found."
    echo "[INFO] Installing huggingface_hub..."
    python3 -m pip install -U huggingface_hub
fi

if [ -d "${MODEL_DIR}" ] && [ "$(find "${MODEL_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "[INFO] CLIP model already exists:"
    echo "       ${MODEL_DIR}"
    echo
    echo "Nothing to download."
    exit 0
fi

mkdir -p "${MODEL_DIR}"

python3 - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${MODEL_DIR}",
)
PY

echo
echo "Location:"
echo "  ${MODEL_DIR}"
echo