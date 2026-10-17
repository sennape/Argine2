@echo off
setlocal

set MODEL=C:\modelli\Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf
set BASE=experiments\EXP-007-router-characterization

echo ============================================================
echo  ARGINE2 EXP-007 - 80B MULTI-TURN V2
echo  DEV + EVAL / 4G,2G,1.5G / repeat 3
echo ============================================================

echo.
echo ===== FREE SPACE BEFORE START =====
powershell -NoProfile -Command "Get-PSDrive C | Select-Object Name,@{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}}"

echo.
echo ============================================================
echo  1/6 - 80B DEV TRACE
echo ============================================================

call argine.bat research trace "%MODEL%" --expert-store --budgets 4G,2G,1.5G --repeat 3 --workload multi-turn-v2 --split dev --out "%BASE%\raw\80b-multi-turn-v2-dev"

if not exist "%BASE%\raw\80b-multi-turn-v2-dev\runs.csv" (
    echo.
    echo ERROR: DEV trace did not complete. runs.csv missing.
    goto :failed
)

echo.
echo ============================================================
echo  2/6 - 80B DEV FIDELITY
echo ============================================================

call argine.bat research fidelity "%BASE%\raw\80b-multi-turn-v2-dev" --out "%BASE%\fidelity\80b-multi-turn-v2-dev"

if not exist "%BASE%\fidelity\80b-multi-turn-v2-dev\fidelity_runs.csv" (
    echo.
    echo ERROR: DEV fidelity_runs.csv missing.
    goto :failed
)

echo.
echo ============================================================
echo  3/6 - COMPRESS DEV
echo ============================================================

compact /c /s:"%BASE%\raw\80b-multi-turn-v2-dev" /i /q
compact /c /s:"%BASE%\fidelity\80b-multi-turn-v2-dev" /i /q

echo.
echo ===== FREE SPACE AFTER DEV =====
powershell -NoProfile -Command "$f=(Get-PSDrive C).Free/1GB; Write-Host ('FreeGB: {0:N2}' -f $f); if($f -lt 20){exit 2}"

if errorlevel 2 (
    echo.
    echo STOP: less than 20 GB free after DEV.
    echo DEV is complete and preserved. EVAL was NOT started.
    goto :lowspace
)

echo.
echo ============================================================
echo  4/6 - 80B EVAL TRACE
echo ============================================================

call argine.bat research trace "%MODEL%" --expert-store --budgets 4G,2G,1.5G --repeat 3 --workload multi-turn-v2 --split eval --out "%BASE%\raw\80b-multi-turn-v2-eval"

if not exist "%BASE%\raw\80b-multi-turn-v2-eval\runs.csv" (
    echo.
    echo ERROR: EVAL trace did not complete. runs.csv missing.
    goto :failed
)

echo.
echo ============================================================
echo  5/6 - 80B EVAL FIDELITY
echo ============================================================

call argine.bat research fidelity "%BASE%\raw\80b-multi-turn-v2-eval" --out "%BASE%\fidelity\80b-multi-turn-v2-eval"

if not exist "%BASE%\fidelity\80b-multi-turn-v2-eval\fidelity_runs.csv" (
    echo.
    echo ERROR: EVAL fidelity_runs.csv missing.
    goto :failed
)

echo.
echo ============================================================
echo  6/6 - COMPRESS EVAL
echo ============================================================

compact /c /s:"%BASE%\raw\80b-multi-turn-v2-eval" /i /q
compact /c /s:"%BASE%\fidelity\80b-multi-turn-v2-eval" /i /q

echo.
echo ============================================================
echo  FINAL RESULTS - DEV
echo ============================================================

type "%BASE%\raw\80b-multi-turn-v2-dev\runs.csv"
type "%BASE%\fidelity\80b-multi-turn-v2-dev\fidelity_runs.csv"

echo.
echo ============================================================
echo  FINAL RESULTS - EVAL
echo ============================================================

type "%BASE%\raw\80b-multi-turn-v2-eval\runs.csv"
type "%BASE%\fidelity\80b-multi-turn-v2-eval\fidelity_runs.csv"

echo.
echo ===== FINAL FREE SPACE =====
powershell -NoProfile -Command "Get-PSDrive C | Select-Object Name,@{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}},@{N='UsedGB';E={[math]::Round($_.Used/1GB,2)}}"

echo.
echo ============================================================
echo  80B DEV + EVAL FINISHED
echo ============================================================
goto :end

:lowspace
echo.
echo DEV completed successfully, but EVAL was skipped for disk safety.
goto :end

:failed
echo.
echo ============================================================
echo  BATCH STOPPED - CHECK THE LAST RUN
echo ============================================================
goto :end

:end
endlocal