@echo off
rem Compatibility entry point for the portable release.
if not exist "%~dp0kataribe.exe" (
  echo [ERROR] kataribe.exe was not found beside this file.
  pause
  exit /b 1
)
start "" "%~dp0kataribe.exe"
exit /b 0
