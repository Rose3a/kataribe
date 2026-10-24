@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Irodori-TTS Initial Setup
call "%~dp0bat\first_setup.bat" %*
set "SETUP_EXIT=%ERRORLEVEL%"
echo.
if "%SETUP_EXIT%"=="0" (
  echo Setup completed successfully.
) else (
  echo Setup failed with exit code %SETUP_EXIT%.
  echo See "%~dp0logs\first-setup.log" for details.
)
pause
exit /b %SETUP_EXIT%
