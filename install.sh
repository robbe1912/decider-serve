#!/usr/bin/env bash
# One-shot sandboxed installer for Linux and macOS. Everything lands in this
# folder: venv/ (deps), .cache/ (pip + kernel caches), models/ (weights).
#   Linux x86_64 + NVIDIA  -> torch 2.5.1 cu121 (+ matching triton via torch)
#   Linux x86_64 / aarch64 without NVIDIA, macOS arm64 -> CPU / MPS torch
# Requires: Python >= 3.10 as python3.12 / python3 / python.
set -euo pipefail
cd "$(dirname "$0")"
export PIP_CACHE_DIR="$PWD/.cache/pip"

PY=""
for c in python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "[install] Python >= 3.10 not found."; exit 1; }

if [ ! -x venv/bin/python ]; then
  echo "[install] creating venv ..."
  "$PY" -m venv venv
fi
VPY="$PWD/venv/bin/python"

ARCH="$(uname -m)"; OS="$(uname -s)"
if [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "[install] Linux x86_64 + NVIDIA -> torch 2.5.1 cu121"
  "$VPY" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
elif [ "$OS" = "Linux" ]; then
  echo "[install] Linux $ARCH without NVIDIA -> CPU torch 2.5.1 (small wheels)"
  "$VPY" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
else
  echo "[install] $OS $ARCH -> PyPI torch 2.5.1 (macOS: MPS wheel)"
  "$VPY" -m pip install torch==2.5.1
fi
"$VPY" -m pip install -r requirements.txt
if [ -f models/decider-2b/config.json ]; then
  "$VPY" smoke_test.py
else
  cat <<'EOF'
[install] deps ready, but models/decider-2b is empty.
Plop the weights in (copy a snapshot's files from huggingface.co/Mapika/...)
or run:  venv/bin/python fetch_models.py
Then verify with:  venv/bin/python smoke_test.py
Start the server with:  ./serve.sh
EOF
fi
