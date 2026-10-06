"""Shared helpers: drop-in model resolution and VRAM checks. No network here.

Models are plain folders inside models/ -- copy a snapshot's files in
(config.json, *.safetensors, tokenizer files, decider_config.json) or use
fetch_models.py once. No hub-cache layout, no downloads at serve time.
"""
import os
from pathlib import Path

SHORTCUTS = {
    "2b": "decider-2b",
    "4b": "decider-4b",
}

GiB = 1024 ** 3
# CUDA context + activation/workspace headroom on top of the bf16 weights.
VRAM_MARGIN_BYTES = int(1.3 * GiB)

# Sandbox: everything the app writes lives inside its own folder.
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

_MISSING = """model folder '{path}' is missing or incomplete (need config.json + *.safetensors).

Either plop the weights in manually -- download a snapshot from
https://huggingface.co/Mapika/{name} and copy its files into models/{name}/ --
or run once:  {py} fetch_models.py"""


def model_dir(spec=None):
    """Resolve '2b' | '4b' | 'decider-2b' | an explicit folder path.

    -> (path, label); raises SystemExit with instructions when the folder
    does not look like a model (config.json + at least one .safetensors).
    """
    spec = spec or os.environ.get("DECIDER_MODEL") or "2b"
    if os.path.isdir(spec):
        path = Path(spec)
    else:
        path = MODELS_DIR / SHORTCUTS.get(spec.lower(), spec)
    ok = (path / "config.json").is_file() and any(path.glob("*.safetensors"))
    if not ok:
        raise SystemExit(_MISSING.format(path=path, name=path.name,
                                         py=sys_executable()))
    return str(path), path.name


def sys_executable():
    import sys
    return sys.executable


def weights_bytes(path):
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))


def cuda_headroom():
    """(free, total) bytes of CUDA device 0; (0, 0) when CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0, 0
        return torch.cuda.mem_get_info()
    except Exception:
        return 0, 0


def fits_on_gpu(path):
    """-> (fits, free_bytes): enough free VRAM for the weights plus margin?"""
    free, _total = cuda_headroom()
    need = weights_bytes(path) + VRAM_MARGIN_BYTES
    return free >= need, free
