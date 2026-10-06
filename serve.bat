@echo off
rem Start the resident decision server (2B on GPU by default).
rem   serve.bat 4b          run the 4B triage model instead (CPU by default)
rem   serve.bat --port 8010 any extra args pass through
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" serve.py %*
