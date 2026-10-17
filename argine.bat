@echo off
REM Avvia Argine senza installarlo: argine.bat <comando> [opzioni]
setlocal
cd /d "%~dp0"
python -m argine %*
endlocal
