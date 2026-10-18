@echo off
setlocal
echo === Irodori TTS bf16 trt ===
echo.
echo Usage examples (run from cmd or double-click):
echo   run.bat --list-speakers
echo   run.bat --speaker unleashguang --text "Hello" --out greetings\out.wav
echo   run.bat --speaker unleashguang --text "Test" --seed 2002 --num-steps 8 --seconds 10
echo.
echo Environment overrides (optional):
echo   IRODORI_BACKEND=trt or torch    force a specific backend
echo   IRODORI_EMBED_DIR=path           search root for *.speaker.safetensors
echo   IRODORI_EMBED_DIRS=root1;root2   additional search roots, ; or , separated
echo   IRODORI_PLAN=path               TensorRT engine plan
echo   IRODORI_RUNTIME_DIR=path        parent of runtime source folders
echo   IRODORI_PYTHON=path             explicit Python interpreter
echo   IRODORI_CHECKPOINT=path         model checkpoint
echo   IRODORI_HF_HOME=path            HuggingFace cache
echo   IRODORI_CACHE_DIR=path          small temp area for active speaker
echo === end of help ===
echo Anchor working directory at the bat's own location so double-click and
echo shell launches both work the same way.
cd /d "%~dp0"
echo Move up one level so ROOT points at the workdir (not wrapper/).
cd ..
set "ROOT=%cd%"
cd

if not defined IRODORI_BACKEND set "IRODORI_BACKEND=auto"
if not defined IRODORI_EMBED_DIR if exist "%ROOT%\embeddings" set "IRODORI_EMBED_DIR=%ROOT%\embeddings"
if not defined IRODORI_PLAN if exist "%ROOT%\bf16-fallback\fallback_bf16.plan" set "IRODORI_PLAN=%ROOT%\bf16-fallback\fallback_bf16.plan"
if not defined IRODORI_RUNTIME_DIR if exist "%ROOT%\runtime" set "IRODORI_RUNTIME_DIR=%ROOT%\runtime"
if not defined IRODORI_CHECKPOINT if exist "%ROOT%\models\v41small.bf16.safetensors" set "IRODORI_CHECKPOINT=%ROOT%\models\v41small.bf16.safetensors"
if not defined IRODORI_HF_HOME set "IRODORI_HF_HOME=%ROOT%\.cache\huggingface"
if not defined IRODORI_CACHE_DIR set "IRODORI_CACHE_DIR=%ROOT%\.cache\irodori"
if not defined IRODORI_PYTHON if exist "%ROOT%\.venv\Scripts\python.exe" set "IRODORI_PYTHON=%ROOT%\.venv\Scripts\python.exe"
rem Defaults are resolved relative to this folder. No developer-machine
rem absolute paths here; set the IRODORI_ variables for assets kept elsewhere.
if not defined IRODORI_PYTHON set "IRODORI_PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not defined IRODORI_RUNTIME_DIR set "IRODORI_RUNTIME_DIR=%ROOT%\runtime"
if not defined IRODORI_EMBED_DIR set "IRODORI_EMBED_DIR=%ROOT%\embeddings"
if not defined IRODORI_PLAN set "IRODORI_PLAN=%ROOT%\bf16-fallback\fallback_bf16.plan"
if not defined IRODORI_CHECKPOINT set "IRODORI_CHECKPOINT=%ROOT%\models\v41small.bf16.safetensors"
if not defined IRODORI_OUT_DIR set "IRODORI_OUT_DIR=%ROOT%\outputs"

set "PYEXE=%IRODORI_PYTHON%"
if not exist "%ROOT%\.venv\Scripts\python.exe" if exist "%ROOT%\setup_venv.bat" (
    call "%ROOT%\setup_venv.bat" --no-pause
    if errorlevel 1 exit /b 1
    set "PYEXE=%ROOT%\.venv\Scripts\python.exe"
)
if not exist "%PYEXE%" (
    echo [error] python.exe not found: %PYEXE%
    exit /b 1
)

if "%~1"=="" (
    "%PYEXE%" "%ROOT%\wrapper\tts_cli.py" --help
    set "RC=%ERRORLEVEL%"
    if not "%RC%"=="0" echo [exit] %RC%
    exit /b %RC%
)

set "SCRIPT=%ROOT%\wrapper\tts_cli.py"
if not exist "%SCRIPT%" (
    echo [error] wrapper script not found: %SCRIPT%
    exit /b 1
)

set "ARGS="
:loop
if "%~1"=="" goto :run
set "ARGS=%ARGS% %~1"
shift
goto :loop

:run
echo [info] backend=%IRODORI_BACKEND%
echo [info] plan=%IRODORI_PLAN%
echo [info] runtime=%IRODORI_RUNTIME_DIR%
echo [info] script=%SCRIPT%
echo [info] args: %ARGS%
"%PYEXE%" "%SCRIPT%" --mode cli --out-dir "%IRODORI_OUT_DIR%" --backend %IRODORI_BACKEND% %ARGS%
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo [exit] %RC%
exit /b %RC%
