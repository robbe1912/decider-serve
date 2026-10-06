"""Compatibility + sandbox shims: import this BEFORE decider/transformers.

0. TRITON_CACHE_DIR is pinned inside the app folder before any triton import,
   so compiled kernels never land in ~/.triton (the default).
1. transformers >= 5.15 calls torch.compiler.is_exporting() in the Qwen3.5
   forward; the attribute only exists on torch >= 2.6. We never export, so False.
2. transformers' use_kernel_func_from_hub_with_fallback picks the fla (Triton)
   gated-delta-rule implementation whenever fla is importable -- device-blind.
   Triton kernels cannot touch CPU tensors, so any CPU model load crashes with
   "Pointer argument cannot be accessed from Triton (cpu tensor?)".
   We re-bind the module-level functions to dispatch on the query device:
   cuda -> fla Triton kernels, anything else -> the pure-torch reference
   implementation that the decorator wrapped (reachable through __wrapped__).
"""
import functools
import os
from pathlib import Path

# --- sandbox: kernel caches stay inside the app folder (before triton import) ---
_CACHE = Path(__file__).resolve().parent / ".cache" / "triton"
os.environ.setdefault("TRITON_CACHE_DIR", str(_CACHE))

import torch.compiler  # noqa: E402  (after the env, on purpose)

if not hasattr(torch.compiler, "is_exporting"):
    torch.compiler.is_exporting = lambda: False


def _device_dispatch_patched():
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    def unwrap_reference(fn):
        ref = fn
        while getattr(ref, "__wrapped__", None) is not None:
            ref = ref.__wrapped__
        return ref if ref is not fn else None

    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(m, name, None)
        ref = unwrap_reference(fn) if fn is not None else None
        if ref is None:
            continue

        @functools.wraps(fn)
        def dispatched(*args, _fn=fn, _ref=ref, **kwargs):
            target = _fn if (args and torch.is_tensor(args[0]) and args[0].is_cuda) else _ref
            return target(*args, **kwargs)

        setattr(m, name, dispatched)


try:
    _device_dispatch_patched()
except Exception as e:  # pragma: no cover - fail loudly at startup, not mid-inference
    raise RuntimeError(f"compat: could not patch Qwen3.5 CPU kernel dispatch: {e}") from e
