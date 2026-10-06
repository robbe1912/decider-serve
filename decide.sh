#!/usr/bin/env bash
# One-off CLI decision -> JSON + journal.jsonl (see decide.py --help).
cd "$(dirname "$0")"
exec venv/bin/python decide.py "$@"
