@echo off
rem One-shot sandboxed installer (Windows x64 + NVIDIA): everything lands inside
rem this folder -- venv\ (deps), .cache\ (pip + kernel caches), models\ (weights).
rem Linux/macOS: use install.sh. Requires: Python 3.12 reachable as py -3.12 or
rem python, and an NVIDIA driver (the torch wheel bundles the CUDA 12.1 runtime).
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0.cache\pip"

set PY=
py -3.12 -c "print(1)" >nul 2>&1 && set PY=py -3.12
if not defined PY ( python -c "print(1)" >nul 2>&1 && set PY=python )
if not defined PY (
    echo [install] Python 3.12 not found. Install it from https://python.org ^(check "Add to PATH"^).
    exit /b 1
)

if not exist venv\Scripts\pip.exe (
    echo [install] creating venv ...
    %PY% -m venv venv || exit /b 1
)
rem Quirk seen on some boxes: venv gets pip.exe but no python.exe. Fix: copy the
rem base interpreter in (venv python.exe finds its home via pyvenv.cfg next to it).
if not exist venv\Scripts\python.exe (
    echo [install] venv python.exe missing - copying base interpreter ...
    for /f "delims=" %%i in ('%PY% -c "import sys;print(sys.executable)"') do copy /y "%%i" venv\Scripts\python.exe >nul || exit /b 1
)

set VPY=%~dp0venv\Scripts\python.exe
echo [install] installing torch 2.5.1 cu121 ...
"%VPY%" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 || exit /b 1
echo [install] installing pinned requirements ...
"%VPY%" -m pip install -r requirements.txt || exit /b 1
echo [install] installing triton-windows (Triton kernels, Windows/NVIDIA only) ...
"%VPY%" -m pip install triton-windows==3.8.0.post29 || exit /b 1

if exist models\decider-2b\config.json (
    echo [install] running smoke test ...
    "%VPY%" smoke_test.py || exit /b 1
    echo [install] done. Start the server with serve.bat, ask questions with decide.bat.
) else (
    echo [install] deps ready, but models\decider-2b is empty.
    echo   Plop the weights in ^(copy a snapshot's files from huggingface.co/Mapika/...^)
    echo   or run:  venv\Scripts\python.exe fetch_models.py
    echo   then verify with:  venv\Scripts\python.exe smoke_test.py
)
