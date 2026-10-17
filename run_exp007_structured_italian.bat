@echo off

setlocal

cd /d "%~dp0"

set "ARGINE_LLAMA_DIR=%CD%\llama-research-v2"

echo ============================================================
echo EXP-007 unattended batch
echo.
echo structured-json + Italian
echo Multi-turn intentionally excluded
echo.
echo ARGINE_LLAMA_DIR=%ARGINE_LLAMA_DIR%
echo ============================================================
echo.

python exp007_batch_runner.py

echo.
echo ============================================================
echo Runner terminato.
echo ============================================================

pause