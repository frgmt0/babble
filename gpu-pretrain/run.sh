#!/usr/bin/env bash
# One-command GPU (or CPU-smoke) pretrain. Run from anywhere:
#   ./run.sh --smoke
#   ./run.sh                         # needs ./start.pt
#   ./run.sh /path/to/latest.pt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v uv >/dev/null 2>&1; then
  echo "[run] installing uv into ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -d .venv ]]; then
  echo "[run] creating .venv"
  uv venv .venv --python 3.12
fi
# shellcheck disable=SC1091
source .venv/bin/activate

SMOKE=0
TRAIN_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--smoke" ]]; then
    SMOKE=1
  elif [[ -f "$arg" && "$arg" == *.pt && "$arg" != --* ]]; then
    TRAIN_ARGS+=(--checkpoint "$arg")
  else
    TRAIN_ARGS+=("$arg")
  fi
done

if [[ "${BABBLE_PRETRAIN_SMOKE:-}" == "1" ]]; then
  SMOKE=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[run] installing torch from the CUDA 12.8 index (Blackwell / 6000)"
  uv pip install -q "torch>=2.6" --index-url https://download.pytorch.org/whl/cu128
else
  echo "[run] no nvidia-smi; installing CPU torch"
  uv pip install -q "torch>=2.1"
fi
uv pip install -q huggingface_hub pyarrow datasets

if [[ "$SMOKE" -eq 1 ]]; then
  TRAIN_ARGS+=(--smoke)
fi

mkdir -p run-output
# Tee console to a plain-text log; training still prints to stdout.
exec python -u "$HERE/train.py" \
  --config "$HERE/config.json" \
  --output-dir "$HERE/run-output" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "$HERE/run-output/train.log"
