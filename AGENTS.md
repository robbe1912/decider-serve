# AGENTS.md — install and verify decider-serve on a fresh machine

Helper for any agent (or human) setting up a working copy of this wrapper
app. Follow top to bottom. The app is fully sandboxed: everything it writes
stays inside its own folder.

## What this is

Local serving stack for Mapika **decider** models (one-pass typed decisions:
state + questions + option lists → calibrated probability distribution, no
text generation). `2b` is the resident GPU router (~85 ms/decide on an RTX
3080); `4b` (revision **v2**, better-calibrated on hard items) is the batch
triage model, CPU by default (~2.6 s/decide). One model per process — never
load 2b and 4b together on a 10 GB card.

**Weights never enter the git repo.** `models/` is gitignored by design;
`git ls-files models/` stays empty. Weight files are fetched from Hugging
Face at setup time (step 2) — nothing is bundled or redistributed here.

## Prerequisites

- Python **3.10–3.12** visible as `python3.12`/`python3` (Linux/macOS) or
  `py -3.12`/`python` (Windows).
- ~15 GB disk: venv ~7 GB + weights 3.5 GB (2b) [+ 7.8 GB (4b, optional)].
- GPU optional. NVIDIA CUDA → Triton kernels; anything else → pure-torch
  reference path (slower, same numbers to fp noise).
- **Windows ARM64: unsupported** (no torch wheels). Stop and report instead.

## 1. Install

```bat
:: Windows x64 — from the repo root
install.bat
```

```bash
# Linux (x64: cu121 if nvidia-smi present, else CPU wheels; aarch64: CPU)
# macOS arm64 (MPS)
bash install.sh
```

The installer creates `venv/`, installs pinned deps (`requirements.txt` +
platform torch), and runs the smoke test once weights are present. All
caches are pinned into `<repo>/.cache/` — nothing lands in the user profile.

Windows gotchas seen in the wild:
- If `venv\Scripts\python.exe` is missing after venv creation, `install.bat`
  copies the base `python.exe` in (a known Windows venv quirk).
- In POSIX-y shells, call the venv python by absolute path
  (`<repo>\venv\Scripts\python.exe`) — the relative form and bare `python`
  have been unreliable.
- To run `.bat` files from such a shell:
  `powershell -NoProfile -Command "& '<repo>\serve.bat' --port 8000"`
  (MSYS `cmd //c` quoting is unreliable).

## 2. Get weights (plain drop-in folders)

`models/decider-2b/` and `models/decider-4b/` are **plain folders** — no HF
hub layout, no network at load time. Either:

- **Fetch (normal path):** `venv\Scripts\python fetch_models.py` (Windows)
  or `venv/bin/python fetch_models.py` (Linux/macOS). Downloads both models
  from Hugging Face into `models/`, pinning 4b to revision `v2` (deliberate:
  better calibration on hard items than the default v2.1).
- **Plop:** copy a snapshot's files from huggingface.co into those folders.
  Each needs `config.json`, `*.safetensors`, `tokenizer.json`,
  `tokenizer_config.json`, `decider_config.json` (4b: use revision `v2`).

## 3. Verify — all four checks must pass

From the repo root; `PY` = the venv python (`venv/Scripts/python.exe` on
Windows, `venv/bin/python` elsewhere).

1. **Smoke test** (loads 2b on GPU or CPU fallback, runs the model-card
   example, then 4b on CPU; skips 4b-GPU if VRAM is short):
   ```
   <PY> smoke_test.py
   ```
   Expect final line `ALL PASS`; canonical example probs topped by
   `billing ≈ 0.91` (2b) / `≈ 0.914` (4b). Reference latency on an RTX
   3080: 2B GPU ~85 ms; 4B CPU ~2.6 s. Exact ms varies by hardware; probs
   must match to ~1e-3.
2. **Sandbox audit** (proves zero footprint outside the folder):
   ```
   <PY> audit_sandbox.py
   ```
   Expect `ALL PASS` (no `~/.triton`, no decider dirs in the user HF cache;
   on Windows additionally: no registry uninstall keys, no user-env edits).
3. **HTTP server round-trip**:
   ```
   <PY> serve.py            # or serve.bat / ./serve.sh; add --port 8000
   # in another shell:
   curl -s http://127.0.0.1:8000/decide -H "Content-Type: application/json" \
     -d '{"context":"My card was charged twice for the same purchase.",
          "schema":{"Which department should handle this?":
                    {"type":"choice","options":["billing","technical support","sales"]}}}'
   ```
   Expect JSON with `billing` dominant (~0.91). `POST /v1/systemone` takes
   `{"state", "questions", ...}` (Jev format) and returns
   choice/certainty/probabilities per question. `GET /health` = readiness.
4. **CLI + journal**:
   ```
   <PY> decide.py --state "My card was charged twice for the same purchase." \
        -q "Which department should handle this?" --opts "billing,technical support,sales"
   ```
   Prints the probs JSON and appends a line to `journal.jsonl`
   (`ts, model, question, options, probs, chosen`). `--device cpu` forces
   CPU; `--url http://127.0.0.1:8000` goes through a running server.

VRAM policy is enforced: `serve.py --model 4b` defaults to CPU; a 4b GPU
load is refused unless ~9.1 GiB is free (override `--force-gpu`, idle card
only). 2b needs ~4.5 GiB and coexists with a running game engine.

## Rules for agents working in this repo

- `import compat` stays the **first** app import in every entry point — it
  shims `torch.compiler.is_exporting` (torch 2.5.1 vs transformers 5.18) and
  rebinds the Qwen3.5 gated-delta kernels to dispatch cuda→fla Triton /
  anything-else→reference. Without it every CPU load crashes inside Triton.
- Do **not** enable `DECIDER_COMPILE=1` / `use_graphs=True`: torch 2.5.1's
  inductor is API-incompatible with the installed triton (`triton_key`
  ImportError). Eager is the supported mode; `serve.py` pins a reduced
  graph-bucket grid via `DECIDER_T_BUCKETS`/`DECIDER_B_BUCKETS` so startup
  stays ~16 s instead of minutes.
- Keep the sandbox invariants: caches pinned (`TRITON_CACHE_DIR` in
  compat.py, `PIP_CACHE_DIR` in installers, `HF_HOME` in fetch_models.py),
  weights only in `models/` and never committed, nothing in
  registry/user-env/global caches. `audit_sandbox.py` is the regression
  check — run it after touching any of this.
- One model per process; `smoke_test.py` already sequences 2b→del→4b.
- Dependency changes go in `requirements.txt` (torch stays 2.5.1-pinned per
  platform in the installers; triton-windows is Windows-only and installed
  by `install.bat`, not requirements.txt — Linux gets triton via the torch
  wheel).

## Troubleshooting (all hit and fixed before)

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: transformers` at `compat` import | requirements step skipped | `<PY> -m pip install -r requirements.txt` |
| `torch.compiler has no attribute is_exporting` | compat not imported first | keep `import compat` before decider/transformers |
| `Pointer argument cannot be accessed from Triton (cpu tensor?)` | dispatch patch not applied | same — import compat first |
| inductor/compile `triton_key` ImportError | torch↔triton matrix | stay eager (rule above) |
| `causal_conv1d` fallback warning | no Windows wheels for that lib | expected; correct but slower, ignore |
| `Python >= 3.10 not found` (install.sh) | minimal container image | install `python3.12 python3.12-venv` first (real distros have it) |
| 4b GPU load refused | VRAM guard | intended; use CPU or `--force-gpu` on an idle card |

Background (why eager, the version matrix, latency tables, sandbox audit
guarantees): `SETUP.md`. User-facing overview: `README.md`. Removing the
app: `uninstall.bat` / `uninstall.sh` (stops processes, sweeps residue,
deletes the folder).
