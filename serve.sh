#!/usr/bin/env bash
# Start the resident decision server (2B on GPU by default).
#   ./serve.sh 4b           run the 4B triage model instead (CPU by default)
#   ./serve.sh --port 8010  any extra args pass through
cd "$(dirname "$0")"
exec venv/bin/python serve.py "$@"
