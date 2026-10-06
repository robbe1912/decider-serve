"""Resident HTTP server for local decider models: a thin launcher around decider.serve.

Loads exactly ONE model per process, chosen by --model or $DECIDER_MODEL:
  2b  -> models/decider-2b (default; the resident GPU router)
  4b  -> models/decider-4b (batch triage; CPU by default)
  or an explicit model folder path. Weights are plain drop-in folders; see
  fetch_models.py for the one-shot fetcher (4b pins revision v2 there).

Device policy for a 10 GB RTX 3080:
  * 2b defaults to cuda, 4b defaults to cpu -- the two are never loaded on the
    card together from this launcher.
  * Before any cuda load, free VRAM is checked against the safetensors size plus
    a margin. If the model does not fit (e.g. another app is holding VRAM),
    the launcher falls back to cpu with a warning, or exits if --force-gpu was
    given. This also stops a second instance from stacking on a resident one.

Endpoints (served by decider.serve): POST /decide, POST /v1/systemone,
GET /v1/models, /health, /stats. Extra tuning via the DECIDER_* env vars
documented in decider/serve.py (DECIDER_TEMPERATURE, DECIDER_MAX_BATCH, ...).
"""
import argparse
import os
import sys
import compat  # torch<2.6 shim: must run before decider/transformers import


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="2b | 4b | models/<name> | explicit folder (default: $DECIDER_MODEL or 2b)")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"],
                    help="default: $DECIDER_DEVICE, else auto -> cuda for 2b, cpu for 4b")
    ap.add_argument("--host", default=os.environ.get("DECIDER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DECIDER_PORT", "8000")))
    ap.add_argument("--force-gpu", action="store_true",
                    help="exit with an error instead of falling back to cpu when VRAM is short")
    args = ap.parse_args()

    from common import GiB, cuda_headroom, fits_on_gpu, model_dir, weights_bytes

    path, label = model_dir(args.model)
    is_4b = "decider-4b" in label
    device = args.device or os.environ.get("DECIDER_DEVICE") or ("cpu" if is_4b else "auto")
    if device == "auto":
        free, _ = cuda_headroom()
        device = "cuda" if free > 0 else "cpu"

    if device.startswith("cuda"):
        fits, free = fits_on_gpu(path)
        need = weights_bytes(path)
        if not fits:
            msg = (f"[serve] {label} needs ~{(need + 1.3 * GiB) / GiB:.1f} GiB free VRAM "
                   f"(weights {need / GiB:.1f} GiB + margin), only {free / GiB:.1f} GiB free")
            if args.force_gpu:
                sys.exit(msg)
            print(msg + " -- falling back to cpu (another VRAM consumer is resident?)",
                  file=sys.stderr)
            device = "cpu"

    # decider.serve reads these at import time; set them before the import.
    os.environ["DECIDER_MODEL"] = path
    os.environ["DECIDER_DEVICE"] = device
    # Startup speed: the default 13x7=89-graph capture grid takes minutes with
    # eager kernels; a short-row router needs only small buckets. Rows longer
    # than the last bucket run eager automatically (decider.serve long-row path).
    os.environ.setdefault("DECIDER_T_BUCKETS", "64,128,256,512,1024")
    os.environ.setdefault("DECIDER_B_BUCKETS", "1,2,4,8")

    import uvicorn
    from decider.serve import app  # noqa: E402  (import after env is set)

    print(f"[serve] model={label} ({path})")
    print(f"[serve] device={device} -> http://{args.host}:{args.port}  (POST /decide, POST /v1/systemone)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
