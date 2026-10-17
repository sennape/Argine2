@echo off
REM EXP-010 STEP 5, generated from selection.json (do not edit by hand).
REM Runs the DEV-selected configuration of each model on its 7 EVAL cells.
REM The adaptive command may also emit mandatory baseline/base rows; only the selected variant enters the primary comparison.
cd /d "%~dp0..\.."
REM 30b: selected e32 (variant adaptive-e32)
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\30b-natural-language-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\30b-natural-language-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\30b-coding-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\30b-coding-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\30b-reasoning-math-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\30b-reasoning-math-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\30b-structured-json-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\30b-structured-json-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\30b-italian-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\30b-italian-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\30b-transition-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\30b-transition-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\30b-multi-turn-v2-eval\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\30b-multi-turn-v2-eval
if errorlevel 1 exit /b 1
REM 80b: selected e32 (variant adaptive-e32)
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\80b-natural-language-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\80b-natural-language-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\80b-coding-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\80b-coding-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\80b-reasoning-math-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\80b-reasoning-math-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\80b-structured-json-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\80b-structured-json-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\80b-italian-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\80b-italian-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\80b-transition-eval-v2\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\80b-transition-eval-v2
if errorlevel 1 exit /b 1
call argine.bat research adaptive experiments\EXP-007-router-characterization\raw\80b-multi-turn-v2-eval\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32 --pinned 0 --no-ablate --out experiments\EXP-010-adaptive-allocation\eval-adaptive\80b-multi-turn-v2-eval
if errorlevel 1 exit /b 1
echo EXP-010 STEP 5 done.
