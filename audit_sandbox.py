"""Sandbox audit: verify nothing leaks outside the app folder. Read-only.

Checks (exit 1 on any FAIL):
  1. No Triton kernel cache under the user profile (compat.py pins
     TRITON_CACHE_DIR to <app>/.cache/triton before triton imports).
  2. No decider model dirs or lock dirs in the user-level Hugging Face cache.
  3. Windows: no uninstall registry keys and no user-environment values
     referencing decider-serve.
  4. Informational: pip download-cache location (internalized by the install
     scripts via PIP_CACHE_DIR; a pre-sandboxing install may have left bytes
     in the OS cache -- uninstall offers an optional purge).
"""
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
fails = []


def check(name, ok, detail=""):
    print(f"[audit] {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def main():
    print(f"[audit] app folder: {APP}")

    # 1. Triton kernel cache must not live in the user profile.
    triton_home = APP / ".cache" / "triton"   # pinned by compat.py at runtime
    user_triton = Path.home() / ".triton"
    leaked = user_triton.exists() and any(user_triton.rglob("*"))
    check("no ~/.triton kernel cache in user profile", not leaked,
          f"cache pinned to {triton_home}" if not leaked else str(user_triton))

    # 2. No decider artifacts in the user-level HF cache (models or locks).
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    leftovers = []
    if hub.is_dir():
        leftovers = [p for p in hub.rglob("models--Mapika--decider-*") if p.exists()]
    check("no decider dirs in user-level HF cache", not leftovers,
          ", ".join(str(p) for p in leftovers))

    # 3. Registry / user environment (Windows only).
    if sys.platform == "win32":
        import winreg
        hits = []
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                base = winreg.OpenKey(root, r"Software\Microsoft\Windows\CurrentVersion\Uninstall")
            except OSError:
                continue
            with base:
                for i in range(winreg.QueryInfoKey(base)[0]):
                    sub = winreg.EnumKey(base, i)
                    if "decider" in sub.lower():
                        hits.append(sub)
        check("no decider uninstall registry keys", not hits, ", ".join(hits))
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env:
                for i in range(winreg.QueryInfoKey(env)[1]):
                    name, value, _ = winreg.EnumValue(env, i)
                    if "decider" in str(value).lower():
                        hits.append(name)
        except OSError:
            pass
        check("no decider-serve user-environment edits", not hits, ", ".join(hits))

    # 4. Informational: pip cache + internal caches.
    internal_triton = APP / ".cache" / "triton"
    print(f"[audit] info  app kernel cache: {internal_triton}"
          + (" (present)" if internal_triton.exists() else " (not created yet -- run smoke_test.py first)"))
    print(f"[audit] info  OS pip cache dir: "
          f"{os.environ.get('PIP_CACHE_DIR', '<pip default: user profile>')}")
    print(f"[audit] info  PIP_CACHE_DIR internalized by install scripts -> "
          f"{APP / '.cache' / 'pip'}")

    print(f"[audit] {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
