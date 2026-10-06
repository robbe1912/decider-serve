@echo off
rem Full uninstall: removes EVERYTHING this app created.
rem   - the whole app folder (venv, models, caches, journal, code, .git)
rem   - external residue from pre-sandboxing runs: ~/.triton kernel cache and
rem     decider lock dirs in the user-level Hugging Face cache (both regenerable)
rem   - optional: purge the shared pip download cache (prompt)
setlocal
cd /d "%~dp0"
for %%Z in ("%~dp0.") do set "TARGET=%%~fZ"
echo This deletes the entire app folder:
echo   %TARGET%
echo (venv, model weights, caches, journal, code, git history)
set /p CONFIRM=Type YES to continue:
if /I not "%CONFIRM%"=="YES" exit /b 1

rem stop any python processes running from this folder (not this cmd/powershell)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*decider-serve*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

rem external residue (idempotent, regenerable caches only)
if exist "%USERPROFILE%\.triton" rmdir /s /q "%USERPROFILE%\.triton"
for /d %%D in ("%USERPROFILE%\.cache\huggingface\hub\.locks\models--Mapika--decider-*") do rmdir /s /q "%%D"
for /d %%D in ("%USERPROFILE%\.cache\huggingface\hub\models--Mapika--decider-*") do rmdir /s /q "%%D"

set /p PIPPURGE=Also purge the SHARED pip download cache in the user profile? (a pre-sandboxing install may have left wheel downloads there; purge forces other projects to re-download) (y/N):
if /I "%PIPPURGE%"=="y" if exist venv\Scripts\python.exe venv\Scripts\python.exe -m pip cache purge

rem self-delete from outside the folder (cmd cannot delete the running bat's home)
(echo @timeout /t 3 /nobreak ^>nul & echo @rmdir /s /q "%TARGET%") > "%TEMP%\decider_uninstall.cmd"
start "" /min cmd /c "%TEMP%\decider_uninstall.cmd"
echo Uninstalled. This window can be closed.
