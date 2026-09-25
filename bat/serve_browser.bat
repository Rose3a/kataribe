@echo off
setlocal
if not exist "%~dp0..\voicevox-editor\.env" goto ensure_env
findstr /b /c:"VITE_APP_NAME=voicevox-irodori" "%~dp0..\voicevox-editor\.env" >nul 2>&1
if not errorlevel 1 goto env_ready
:ensure_env
if not exist "%~dp0..\voicevox-editor\.env" type nul > "%~dp0..\voicevox-editor\.env"
findstr /b /c:"VITE_APP_NAME=" "%~dp0..\voicevox-editor\.env" >nul 2>&1
if errorlevel 1 echo VITE_APP_NAME=voicevox-irodori>> "%~dp0..\voicevox-editor\.env"
:env_ready
cd /d "%~dp0..\voicevox-editor"
if not exist "%~dp0..\logs" mkdir "%~dp0..\logs"
set "NODE=%~dp0..\.local\node\24.11.1\node.exe"
if exist "%NODE%" goto node_ready
set "NODE=node.exe"
where node.exe >nul 2>&1
if errorlevel 1 (
  echo Node.js was not found. Install Node.js 24.11+ and retry.
  exit /b 1
)
:node_ready
set "VITE_TARGET=browser"
set "npm_package_name=voicevox-irodori"
rem vite.config.ts reads the app version from npm_package_version, which only
rem exists when pnpm/npm starts vite. Read it from package.json here for the
rem direct node launch so the title always matches the package version.
for /f "usebackq delims=" %%v in (`call "%NODE%" -p "require('./package.json').version"`) do set "npm_package_version=%%v"
if not defined npm_package_version (
  echo Could not read the version from voicevox-editor\package.json.
  exit /b 1
)
"%NODE%" "%~dp0..\voicevox-editor\node_modules\vite\bin\vite.js" --host 127.0.0.1 --port 5173 --strictPort > "%~dp0..\logs\browser-ui.log" 2>&1
exit /b %errorlevel%
