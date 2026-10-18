@echo off
rem Entry point for kataribe tts_cli.py. All arguments are passed through.
rem   bat\tts_cli.bat --list-speakers
rem   bat\tts_cli.bat --speaker tsukuyomi --text-file lines.txt --play
rem   bat\tts_cli.bat --help
rem With no arguments it prints the help. Output goes to kataribe\outputs by default.
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "BOX=%~dp0.."
for %%I in ("%BOX%") do set "BOX=%%~fI"
set "PYTHON=%BOX%\.local\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%BOX%\irodori-tts\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python environment not found. Run setup.bat first.
  exit /b 2
)
set "IRODORI_RUNTIME_DIR=%BOX%\runtime"
set "IRODORI_TRT_LAB_DIR=%BOX%\runtime\trt-lab"
set "IRODORI_CHECKPOINT=%BOX%\models\model.safetensors"
set "IRODORI_HF_HOME=%BOX%\.cache\huggingface"
set "IRODORI_EMBED_DIR=%BOX%\speakers"
set "IRODORI_OUT_DIR=%BOX%\outputs"
cd /d "%BOX%"
if "%~1"=="" (
  "%PYTHON%" "%BOX%\irodori-tts\wrapper\tts_cli.py" --help
  exit /b %errorlevel%
)
"%PYTHON%" "%BOX%\irodori-tts\wrapper\tts_cli.py" %*
exit /b %errorlevel%
