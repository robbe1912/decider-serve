# SETUP — what actually works (2026-10-06, Windows 10 26200, RTX 3080 10 GB)

The short version: `install.bat` (Windows) or `install.sh` (Linux/macOS)
reproduces everything below. This is the long version, battles included.

## Final dependency list (all inside `venv/`, nothing system-wide)

| package | version | notes |
|---|---|---|
| torch | 2.5.1+cu121 | from https://download.pytorch.org/whl/cu121 (2.5.x line proven on this box) |
| transformers | 5.18.0 | decider needs v5 |
| flash-linear-attention | 0.5.2 (+ fla-core 0.5.2) | Triton kernels for the gated-delta-net layers |
| triton-windows | 3.8.0.post29 | Triton for native Windows (github.com/woct0rdho/triton-windows); Windows-only, installed by install.bat — Linux pulls torch-matching triton with the cu121 wheel |
| decider-ai[serve] | 1.8.1 | ≥1.4.0 required (temperature_by_type); serve extra = fastapi/uvicorn/httpx |

Weights are plain drop-in folders inside `models/` (gitignored): copy a snapshot's
files to `models/<name>/` or run `fetch_models.py` once (pins HF_HOME into `.cache/`):

| model | revision | size | role |
|---|---|---|---|
| Mapika/decider-2b | default (v11) | 3.5 GiB | resident GPU router |
| Mapika/decider-4b | **v2** (deliberate: better hard-item calibration than default v2.1) | 7.8 GiB | CPU batch triage |

## Install steps as actually executed

```bat
cd /d <repo>   :: the decider-serve clone
set PIP_CACHE_DIR=<repo>\.cache\pip   :: keep downloads in the app
C:\Python312\python.exe -m venv venv
::: quirk on this box: venv\Scripts gets pip.exe but no python.exe
copy /y C:\Python312\python.exe venv\Scripts\python.exe
venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install triton-windows==3.8.0.post29
venv\Scripts\python.exe fetch_models.py
```

Another quirk of this machine's POSIX-bash tooling: **relative** `venv/Scripts/python.exe`
fails with "command not found" even after `cd` — always invoke it by absolute path.
(The .bat launchers do.)

## Battle 1: torch.compiler.is_exporting (fixed by compat.py)

torch 2.5.1 + transformers 5.18: the Qwen3.5 forward calls
`torch.compiler.is_exporting()`, which only exists on torch ≥ 2.6. Under decider's
compiled Engine this is a hard `InternalTorchDynamoError`. `compat.py` adds
`torch.compiler.is_exporting = lambda: False` (guarded, only when missing) and is
imported first by serve.py / decide.py / smoke_test.py.

## Battle 2: Triton ↔ torch version matrix (decision: eager, native)

- `torch.compile`/inductor on torch 2.5.1 speaks triton 3.1-era API; triton-windows
  3.8 is far newer → `ImportError: cannot import name 'triton_key'`. Matching a
  triton-windows old enough for torch 2.5.1 would break fla 0.5.2, which is built
  for recent triton. Upgrading torch to ≥2.8/cu126 was the alternative; not taken —
  see decision below.
- **Decision: run eager, stay native Windows.** decider's own serving stack defaults
  to `DECIDER_COMPILE=0` anyway (compile-off is the upstream default); our local
  callers pass `use_graphs=False`. The fla **Triton kernels still run** on GPU —
  only inductor fusion/CUDA-graph capture is off. WSL2 was not needed; nothing in
  the stack requires Linux.
- Side effect: `causal_conv1d` is not installed (no Windows wheels); transformers
  uses its reference implementation for the conv part. Correct, slightly slower.

## Battle 3: CPU loads crashed on Triton (fixed by compat.py)

transformers' `use_kernel_func_from_hub_with_fallback` picks the fla Triton
implementation whenever fla is importable — **device-blind**. Any CPU load died with
`ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`.
`compat.py` re-binds `torch_{chunk,recurrent}_gated_delta_rule` in
`transformers.models.qwen3_5.modeling_qwen3_5` to dispatch on the query device:
cuda → fla Triton, cpu → the pure-torch reference (recovered from the decorator's
`__wrapped__` chain). This is what makes the 4B CPU batch path work at all.


**Server startup**: decider's engine captures a (T-bucket × B-bucket) CUDA-graph grid at
startup. The upstream default grid (13×7 = 89 graphs) took **minutes** under eager
kernels; `serve.py` therefore defaults `DECIDER_T_BUCKETS=64,128,256,512,1024` and
`DECIDER_B_BUCKETS=1,2,4,8` → **20 graphs in ~16 s**, ready in well under a minute.
Rows longer than 1024 tokens or batches >8 still work — they run eager per request.

## Measured performance (smoke_test.py, model-card example, 1 question)

| config | load | decide latency (mean / min) | probs |
|---|---|---|---|
| 2B GPU (eager, fla Triton) | 4–7 s | **87 ms / 66 ms** | billing 0.91 ✓ |
| 4B v2 CPU (reference kernels) | 13–15 s | **2.6 s / 2.5 s** (first-ever run: 3.4 s / 2.9 s) | billing 0.914 ✓ |
| 4B v2 GPU (eager, fla Triton) | ~30 s | **181 ms / 176 ms** (peak alloc 8.2 GiB; ran with reduced 0.4 GiB margin because the desktop holds ~1.2 GiB — coexists with nothing) | billing 0.914 ✓ |

The card's reference (~0.93 billing) was measured with compile+graphs; our eager
numbers land at 0.91 — same decision, mildly different calibration readout.

## Run commands

```bat
serve.bat                :: 2B on GPU, http://127.0.0.1:8000
serve.bat 4b             :: 4B triage model, CPU (no GPU needed)
decide.bat --state "..." -q "Which department?" --opts billing,support,sales
<repo>\venv\Scripts\python.exe smoke_test.py
```

VRAM policy: the launcher checks **free** VRAM (weights + 1.3 GiB margin) before
any CUDA load; on shortfall it falls back to CPU with a warning (or exits with
`--force-gpu`). One model per process: 2B and 4B are never loaded together.

## Platforms & sandbox (validated 2026-10-06)

- installers per platform: `install.bat` (Win x64 + NVIDIA, tested here),
  `install.sh` (Linux x64 cu121 when `nvidia-smi` exists, else CPU/MPS wheels;
  macOS arm64 via PyPI). Windows ARM64 unsupported (no torch wheels).
  Cross-check: the Linux CPU path was exercised end-to-end in an
  ubuntu:24.04 docker container (install.sh + decide.py CPU round-trip +
  audit_sandbox.py — all pass, probs match Windows to fp noise).
- `compat.py` pins `TRITON_CACHE_DIR` to `<app>/.cache/triton` **before** any
  triton import; installers pin `PIP_CACHE_DIR`; `fetch_models.py` pins
  `HF_HOME`. Result (audit_sandbox.py, ALL PASS): no registry keys, no
  user-environment edits, no `~/.triton`, no decider dirs in the user-level HF
  cache. Pre-sandboxing runs on this box had leaked `~/.triton` (73 MB) and
  stale HF lock dirs — migrated/deleted; the OS pip cache may still hold wheel
  downloads from that first install (uninstall offers an optional purge).
- full teardown: `uninstall.bat` / `uninstall.sh` — stops app processes,
  removes external residue, deletes the folder.
