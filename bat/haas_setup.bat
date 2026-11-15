@echo off
title kataribe Haas-ja setup (speaker training)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\tools\setup_haas.ps1" %*
set "HAAS_SETUP_EXIT=%errorlevel%"
echo.
if "%HAAS_SETUP_EXIT%"=="0" (
  echo Haas-ja setup completed successfully.
) else (
  echo Haas-ja setup did not complete. See logs\haas-setup.log.
)
echo This window stays open so you can read the result.
pause
exit /b %HAAS_SETUP_EXIT%
