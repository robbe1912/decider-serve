# decider-serve

A fully self-contained local app around the [Mapika decider](https://github.com/Mapika/decider)
decision models: one-pass typed decisions (state + questions + option lists →
calibrated probability distribution, no text generation). Use it as a
dev-workflow router, test-failure triage, or any "pick one of N" call your
tooling needs.

**Fully sandboxed:** everything the app does stays inside its own folder —
the Python venv (`venv/`), all caches (`.cache/`: pip, Triton kernels, HF),
and the model weights (`models/`). No registry keys, no AppData, no user-level
package caches, no system installs. `audit_sandbox.py` verifies exactly that,
and `uninstall.bat` / `uninstall.sh` remove every trace (folder + external
residue) — see [Sandbox & uninstall](#sandbox--uninstall).

## Platform support

| OS | arch | accelerators | installer | kernels | status |
|---|---|---|---|---|---|
| Windows | x64 | NVIDIA CUDA | `install.bat` | fla Triton (triton-windows) | **tested** (RTX 3080, torch 2.5.1 cu121) |
| Linux | x64 | NVIDIA CUDA | `install.sh` | fla Triton (torch-bundled triton) | best-effort (untested here) |
| Linux | x64 / aarch64 | none | `install.sh` | pure-torch reference | **tested** (ubuntu 24.04 container: install + decide + audit pass) |
| macOS | arm64 | MPS | `install.sh` | pure-torch reference | best-effort |
| Windows | arm64 | — | — | — | unsupported (no torch wheels) |

## Install once

```bat
install.bat        (Windows x64 + NVIDIA)
./install.sh       (Linux / macOS)
```

Creates `venv/` with pinned deps (torch 2.5.1, transformers 5.18,
decider-ai 1.8.1, flash-linear-attention 0.5.2; triton-windows on Windows)
and pins every cache into `.cache/`. Requires Python ≥ 3.10 (`py -3.12` /
`python3`) and, for the GPU path, an NVIDIA driver (CUDA runtime ships in the
wheel).

## Models: plop the weights in

`models/` holds one plain folder per model — no cache layout, no magic:

```
models/decider-2b/    config.json  model.safetensors  tokenizer.json  decider_config.json ...
models/decider-4b/    (same shape)
```

Either copy a snapshot's files in manually from
`huggingface.co/Mapika/decider-2b` / `decider-4b`, or run the fetcher once:

```bash
venv/Scripts/python.exe fetch_models.py      # or venv/bin/python on Linux/macOS
```

`decider-4b` is pinned to revision **v2** (better calibration on hard items
than the default v2.1 — model card, "Which version to use"); if you copy
manually, grab the v2 files. Then verify:

```bash
venv/Scripts/python.exe smoke_test.py        # CUDA check + known-answer + latencies
```

## Run

```bat
serve.bat                 :: 2B on GPU (default resident router) on 127.0.0.1:8000
serve.bat 4b              :: 4B triage model (CPU by default — see VRAM policy)
serve.bat --port 8010     :: any serve.py flag passes through      (serve.sh on Linux/macOS)
```

Ask things:

```bat
decide.bat --state "My card was charged twice for the same purchase." ^
  -q "Which department should handle this?" --opts billing,technical support,sales
```

prints the JSON probabilities and appends them to `journal.jsonl`
(timestamp, model, question, options, probs, chosen — the calibration
feedback loop).

Or talk HTTP (endpoints provided by `decider.serve`):

```bash
curl -s http://127.0.0.1:8000/decide -H "Content-Type: application/json" -d '{
  "context": "My card was charged twice for the same purchase.",
  "schema": {"Which department should handle this?":
             {"type": "choice", "options": ["billing", "technical support", "sales"]}}}'
# -> {"Which department should handle this?":
#      {"choice": "billing", "confidence": 0.91, "probabilities": {...}}}
```

`POST /v1/systemone` takes the TypeSafe/Jev format (`{"state", "questions"}`);
`GET /health`, `/stats`, `/v1/models` for ops.

## VRAM policy (10 GB card)

| model | default device | why |
|---|---|---|
| 2B (3.5 GB bf16) | GPU | resident router; coexists with a Godot editor holding VRAM |
| 4B v2 (7.8 GB bf16) | CPU | GPU-only when the card is otherwise idle: weights + context ≈ 8.7 GB of 10 |

The launcher checks **free** VRAM before every CUDA load and falls back to CPU
with a warning (or exits with `--force-gpu`). One model per process; starting
the 4B never stacks on a resident 2B. The 4B runs entirely in system RAM —
`serve.bat 4b` needs no GPU at all.

## Sandbox & uninstall

Guarantees (validated by `audit_sandbox.py`, currently ALL PASS on the dev box):

- **No registry keys, no user-environment edits** (checked live via `winreg`).
- **No Triton kernel cache in the user profile** — `compat.py` pins
  `TRITON_CACHE_DIR` to `<app>/.cache/triton` before triton ever imports.
- **No decider artifacts in the user-level Hugging Face cache** — weights are
  app-local plain folders; `fetch_models.py` pins `HF_HOME` into `.cache/`.
- **No pip downloads into the OS cache** — installers set
  `PIP_CACHE_DIR=<app>/.cache/pip`. (An install made *before* this sandboxing
  may have left wheel downloads in the OS pip cache; uninstall offers an
  optional purge.)

```bat
uninstall.bat      (Windows)         ./uninstall.sh   (Linux/macOS)
```

stops any process running from the folder, removes pre-sandboxing residue
(`~/.triton`, decider lock dirs in the HF cache), optionally purges the
shared pip cache, then deletes the entire app folder.

## Repo layout

```
install.bat / install.sh   sandboxed installers (per platform)
serve.bat / serve.sh       start the HTTP server (2B GPU default)
decide.bat / decide.sh     one-off CLI decision -> JSON + journal.jsonl
uninstall.bat / uninstall.sh  remove every trace
serve.py                   launcher around decider.serve (VRAM guard, shortcuts)
decide.py                  CLI implementation
smoke_test.py              CUDA + load + decide round-trip + latency report
fetch_models.py            optional one-shot weight fetcher into models/
audit_sandbox.py           verifies nothing leaks outside the app folder
common.py                  drop-in model resolution + VRAM checks (no network)
compat.py                  torch 2.5.1 <-> transformers 5.18 shims + cache pinning
AGENTS.md                 runbook: fresh-machine install + verification
SETUP.md                   the actual working recipe, deviations and all
journal.jsonl              append-only decision log
models/                    weights (gitignored, drop-in plain folders)
venv/  .cache/             sandbox (gitignored)
```

## License

Apache-2.0 (this repo). Depends on
[decider](https://github.com/Mapika/decider) (Apache-2.0),
[transformers](https://github.com/huggingface/transformers) (Apache-2.0),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT)
and [triton-windows](https://github.com/woct0rdho/triton-windows) (MIT).
Model weights are fetched by the user from Hugging Face and are **not**
redistributed here.
