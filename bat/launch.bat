@echo off
rem Start the bundled editor. In the installed and the portable layout the editor
rem lives in editor\ (this file used to look for it next to the folder root, which
rem no layout ever used).
setlocal EnableExtensions DisableDelayedExpansion
set "EDITOR=%~dp0..\editor\kataribe.exe"
if not exist "%EDITOR%" (
  echo [ERROR] editor\kataribe.exe was not found.
  echo Keep this file inside the kataribe folder.
  pause
  exit /b 1
)
start "" "%EDITOR%"
exit /b 0
