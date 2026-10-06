"""Fetch model weights into models/ as plain folders (optional, run once).

  python fetch_models.py          -> 2b + 4b (4b pinned to revision v2)
  python fetch_models.py 2b       -> just one

Manual alternative (no script): download a snapshot from
https://huggingface.co/Mapika/<name> and copy its files into models/<name>/
(config.json, *.safetensors, tokenizer files, decider_config.json).
All caches stay inside this folder (HF_HOME is pinned to .cache/).
"""
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(APP / ".cache" / "huggingface"))  # before hub import

# "4b" pins revision v2 deliberately: better calibration on hard items than
# the default v2.1 (model card, "Which version to use").
MODELS = {
    "2b": ("Mapika/decider-2b", None),
    "4b": ("Mapika/decider-4b", "v2"),
}


def main(argv):
    wanted = [a.lower() for a in argv] or list(MODELS)
    unknown = [a for a in wanted if a not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(unknown)} -- choose from {list(MODELS)}")
    from huggingface_hub import snapshot_download
    for key in wanted:
        repo, rev = MODELS[key]
        dest = APP / "models" / repo.split("/")[-1]
        print(f"[fetch] {repo}@{rev or 'main'} -> {dest}", flush=True)
        snapshot_download(repo_id=repo, revision=rev, local_dir=str(dest))
    print("[fetch] done. Verify with: python smoke_test.py")


if __name__ == "__main__":
    main(sys.argv[1:])
