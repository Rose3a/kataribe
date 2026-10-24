@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Use a relative script path after entering the quoted project root.
pushd "%~dp0.."
if errorlevel 1 exit /b 1
if not exist "logs" mkdir "logs"
if not exist "logs" (
  popd
  exit /b 1
)
echo Setup is running. Log: "%CD%\logs\first-setup.log"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\tools\setup.ps1" %*
set "SETUP_EXIT=%errorlevel%"
popd
if not "%SETUP_EXIT%"=="0" pause
exit /b %SETUP_EXIT%
