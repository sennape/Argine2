@echo off
REM ============================================================
REM  Argine - installazione su Windows (Anaconda Prompt o cmd)
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo [1/3] Verifico Python...
python --version 1>nul 2>nul
if errorlevel 1 (
    echo    ERRORE: "python" non trovato nel PATH.
    echo    Apri l'Anaconda Prompt oppure aggiungi Python al PATH e riprova.
    exit /b 1
)
python --version

REM Il progetto si chiamava picostar16. Se la vecchia installazione e' ancora
REM presente, il comando "picostar" resta nel PATH e punta a questa stessa
REM cartella, dove il pacchetto "picostar" non esiste piu': fallirebbe con un
REM ImportError che non dice perche'. Si rimuove in silenzio, se c'e'.
python -m pip uninstall -y picostar16 1>nul 2>nul

echo.
echo [2/3] Installo Argine e le dipendenze...
python -m pip install --upgrade pip 1>nul
python -m pip install -e .
if errorlevel 1 (
    echo    ERRORE durante l'installazione.
    exit /b 1
)

echo.
echo [3/3] Scarico llama.cpp precompilato (CPU + Vulkan)...
python -m argine setup --backend both
if errorlevel 1 (
    echo.
    echo    ATTENZIONE: il runtime llama.cpp non e' pronto. Il messaggio qui
    echo    sopra dice perche': se nomina ARGINE_LLAMA_DIR o PICOSTAR_LLAMA_DIR,
    echo    il problema e' quella variabile, non la rete, e ritentare il
    echo    download non servirebbe. Altrimenti riprova con:
    echo        argine setup
)

echo.
echo ============================================================
echo  Installazione completata. Prova:
echo      argine probe
echo      argine inspect C:\modelli\Qwen3-8B-Q4_K_M.gguf
echo      argine run C:\modelli\Qwen3-8B-Q4_K_M.gguf --auto
echo ============================================================
endlocal
