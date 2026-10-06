@echo off
rem CLI decision: prints JSON probabilities and appends to journal.jsonl.
rem   decide.bat --state "My card was charged twice." -q "Which department?" --opts billing,support,sales
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" decide.py %*
