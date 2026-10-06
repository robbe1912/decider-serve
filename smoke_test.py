"""End-to-end smoke test: CUDA availability, model loads, one decide() round-trip
with the model card's known example, and latency measurements.

  2B on GPU  (falls back to CPU with a warning when VRAM is short, e.g. Godot)
  4B v2 on CPU  (the batch-triage path)
  4B v2 on GPU, only when the 2B is unloaded AND enough VRAM is free

Exit code 0 = all assertions passed (billing dominant in the card example).
"""
import sys
import time
import compat  # torch<2.6 shim: must run before decider/transformers import

from common import GiB, cuda_headroom, fits_on_gpu, model_dir, weights_bytes

STATE = "My card was charged twice for the same purchase."
QUESTIONS = [{"question": "Which department should handle this?",
              "options": ["billing", "technical support", "sales"]}]
EXPECT_TOP = "billing"


def run(label, path, device, reps=5):
    from decider.infer import Decider
    t0 = time.perf_counter()
    d = Decider(path, device=device, use_graphs=False)  # eager: torch 2.5.1 inductor != triton-windows 3.8 API
    load_s = time.perf_counter() - t0
    d.decide(STATE, QUESTIONS)                      # warmup (graph capture etc.)
    out, times = None, []
    for _ in range(reps):
        t = time.perf_counter()
        out = d.decide(STATE, QUESTIONS)
        times.append((time.perf_counter() - t) * 1000)
    probs = out[0]["probs"]
    top = max(probs, key=probs.get)
    ok = top == EXPECT_TOP and out[0]["choice"] == EXPECT_TOP
    print(f"[smoke] {label} device={device} load={load_s:.1f}s "
          f"latency mean={sum(times)/len(times):.1f}ms min={min(times):.1f}ms")
    print(f"[smoke] {label} probs={ {k: round(v, 3) for k, v in probs.items()} } top={top} -> "
          + ("PASS" if ok else "FAIL"))
    return d, ok


def main():
    failures = 0

    import torch
    print(f"[smoke] torch {torch.__version__} cuda_available={torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" device={torch.cuda.get_device_name(0)}")
    else:
        print()
    try:
        import fla  # noqa: F401
        import triton  # noqa: F401
        print(f"[smoke] fla={getattr(fla, '__version__', '?')} triton={triton.__version__} importable")
    except Exception as e:
        print(f"[smoke] fla/triton NOT importable ({e}) -- running without Triton kernels (severalx slower)")

    p2b, label2b = model_dir("2b")
    print(f"[smoke] 2b model dir: {label2b} ({weights_bytes(p2b)/GiB:.1f} GiB) {p2b}")
    free, total = cuda_headroom()
    device_2b = "cuda" if free > 0 else "cpu"
    if device_2b == "cuda":
        fits, fr = fits_on_gpu(p2b)
        if not fits:
            print(f"[smoke] only {fr/GiB:.1f} GiB VRAM free -- running 2B on CPU this time")
            device_2b = "cpu"
    d, ok = run(f"2B", p2b, device_2b)
    failures += not ok

    del d  # never keep 2B and 4B resident together on the 10 GB card
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    p4b, label4b = model_dir("4b")
    print(f"[smoke] 4b model dir: {label4b} ({weights_bytes(p4b)/GiB:.1f} GiB) {p4b}")
    d4, ok4 = run("4B", p4b, "cpu", reps=3)
    failures += not ok4
    del d4

    fits, fr = fits_on_gpu(p4b)
    if fits:
        d4g, ok4g = run("4B", p4b, "cuda", reps=5)
        failures += not ok4g
        del d4g
    else:
        print(f"[smoke] 4B on GPU skipped: needs ~{(weights_bytes(p4b))/GiB + 1.3:.1f} GiB free, {fr/GiB:.1f} GiB free")

    print(f"[smoke] {'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
