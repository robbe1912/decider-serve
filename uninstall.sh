#!/usr/bin/env bash
# Full uninstall: removes EVERYTHING this app created.
#   - the whole app folder (venv, models, caches, journal, code, .git)
#   - external residue from pre-sandboxing runs: ~/.triton kernel cache and
#     decider lock dirs in the user-level Hugging Face cache (both regenerable)
#   - optional: purge the shared pip download cache (prompt)
set -euo pipefail
APP="$(cd "$(dirname "$0")" && pwd)"
echo "This deletes the entire app folder:"
echo "  $APP"
echo "(venv, model weights, caches, journal, code, git history)"
read -r -p "Type YES to continue: " CONFIRM
[ "$CONFIRM" = "YES" ] || exit 1

# stop python processes running from this folder (not this shell)
pkill -f "venv/bin/python.*decider-serve" 2>/dev/null || true
pkill -f "$APP/venv/bin/python" 2>/dev/null || true

# external residue (idempotent, regenerable caches only)
rm -rf "$HOME/.triton"
rm -rf "$HOME"/.cache/huggingface/hub/.locks/models--Mapika--decider-* 2>/dev/null || true
rm -rf "$HOME"/.cache/huggingface/hub/models--Mapika--decider-* 2>/dev/null || true

read -r -p "Also purge the SHARED pip download cache? (a pre-sandboxing install may have left wheel downloads there; purge forces other projects to re-download) (y/N) " PIPPURGE
if [ "${PIPPURGE:-n}" = "y" ] && [ -x "$APP/venv/bin/python" ]; then
  "$APP/venv/bin/python" -m pip cache purge || true
fi

cd /
rm -rf "$APP"
echo "Uninstalled."
