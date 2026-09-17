param([ValidateSet('auto','cuda','cpu','radeon')][string]$Backend='auto', [switch]$Check, [switch]$Force)
$ErrorActionPreference='Stop'
$box = Split-Path $PSScriptRoot -Parent
$stateDir = Join-Path $box '.local'
$python = Join-Path $stateDir 'venv\Scripts\python.exe'
$stateFile = Join-Path $stateDir 'setup.json'
$frontendAsset = Join-Path $box 'voicevox-editor\node_modules\@quasar\extras\material-icons\material-icons.css'
$prebuiltEditor = Join-Path $box 'editor\kataribe.exe'

function Get-Sha256([string]$Path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [IO.File]::OpenRead($Path)
        try { return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '') }
        finally { $stream.Dispose() }
    } finally { $sha.Dispose() }
}

function Test-FrontendInstall {
    if (Test-Path -LiteralPath $prebuiltEditor -PathType Leaf) { return $true }
    # Vite can start even when pnpm's links are incomplete, then fails only once
    # the browser imports the missing package.  Check an asset used by main.ts.
    return Test-Path -LiteralPath $frontendAsset -PathType Leaf
}
$gpu = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Parsec|Remote Display|Hyper-V' } | Select-Object -ExpandProperty Name)
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (!$smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = Get-Command "$env:SystemRoot\System32\nvidia-smi.exe" }
if ($smi) {
    $names = @(& $smi.Source --query-gpu=name --format=csv,noheader 2>$null)
    if ($LASTEXITCODE -eq 0 -and $names.Count) { $gpu = $names }
}
if ($Backend -eq 'auto') {
    if (($gpu -join ' ') -match 'NVIDIA|GeForce|Quadro|Tesla') { $Backend='cuda' }
    elseif (($gpu -join ' ') -match 'AMD|Radeon') { $Backend='radeon' }
    else { $Backend='cpu' }
}
$identity = "$env:COMPUTERNAME|$box|$($gpu -join ',')|$Backend|setup-v2"
$editorDir = Join-Path $box 'voicevox-editor'
$editorEnv = Join-Path $editorDir '.env'
$editorEnvExample = Join-Path $editorDir '.env.example'
if (!(Test-Path $editorEnv) -and (Test-Path $editorEnvExample)) {
    Copy-Item -LiteralPath $editorEnvExample -Destination $editorEnv
    Write-Host '[EDITOR] Created voicevox-editor\.env from .env.example.'
}
if ($Check) { Write-Output "GPU: $($gpu -join ', ')`nBackend: $Backend`nPython: $python"; exit 0 }
New-Item -ItemType Directory -Force $stateDir,(Join-Path $box 'logs') | Out-Null
$lock = $null
try {
    # File lock is released by Windows even when setup is cancelled.
    while (!$lock) {
        try { $lock = [IO.File]::Open((Join-Path $stateDir 'setup.lock'),'OpenOrCreate','ReadWrite','None') }
        catch [IO.IOException] { Write-Host 'Waiting for the other setup to finish...'; Start-Sleep -Seconds 2 }
    }
    if (!$Force -and (Test-Path $stateFile) -and (Test-Path $python) -and (Test-FrontendInstall)) {
        $saved = Get-Content $stateFile -Raw | ConvertFrom-Json
        if ($saved.identity -eq $identity) { exit 0 }
    }
    Start-Transcript -Path (Join-Path $box 'logs\first-setup.log') -Append | Out-Null
    Write-Host "Irodori-TTS first setup / repair`nGPU: $($gpu -join ', ')`nBackend: $Backend`nThis downloads Python, dependencies and models. Please keep this window open."
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $stateDir 'python'
    $env:UV_CACHE_DIR = Join-Path $stateDir 'uv-cache'
    $env:HF_HOME = Join-Path $box '.cache\huggingface'
    $env:PYTHONUTF8 = '1'
    $uvVersion = '0.9.7'
    $uvInstallDir = Join-Path $stateDir 'bin'
    $uv = Join-Path $uvInstallDir 'uv.exe'
    if (!(Test-Path -LiteralPath $uv -PathType Leaf)) {
        Write-Host "[UV] Installing uv $uvVersion into $uvInstallDir..." -ForegroundColor Cyan
        $env:UV_INSTALL_DIR = $uvInstallDir
        $env:UV_NO_MODIFY_PATH = '1'
        $installerUrl = "https://astral.sh/uv/$uvVersion/install.ps1"
        try {
            Invoke-Expression (Invoke-RestMethod -Uri $installerUrl -ErrorAction Stop)
        }
        catch {
            throw "Failed to install uv $uvVersion from $installerUrl. Install uv manually from https://docs.astral.sh/uv/ and run setup again. $($_.Exception.Message)"
        }
        if (!(Test-Path -LiteralPath $uv -PathType Leaf)) {
            throw "uv installer completed but $uv was not created."
        }
    }
    $script:step = 0
    function Run-Uv([string[]]$Arguments) {
        $script:step++
        $started = Get-Date
        Write-Host "[$script:step] START $($Arguments -join ' ')" -ForegroundColor Cyan
        & $uv @Arguments
        if ($LASTEXITCODE -ne 0) { throw "uv failed ($LASTEXITCODE): $($Arguments -join ' ')" }
        Write-Host "[$script:step] DONE  elapsed=$([int]((Get-Date)-$started).TotalSeconds)s" -ForegroundColor Green
    }
    function Ensure-PortableNode {
        $version = '24.11.1'
        $nodeDir = Join-Path $stateDir "node\$version"
        $nodeExe = Join-Path $nodeDir 'node.exe'
        if (Test-Path $nodeExe) { return $nodeDir }
        $archiveName = "node-v$version-win-x64.zip"
        $baseUrl = "https://nodejs.org/dist/v$version"
        $downloadDir = Join-Path $stateDir 'downloads'
        $archivePath = Join-Path $downloadDir $archiveName
        $sumsPath = Join-Path $downloadDir 'SHASUMS256.txt'
        $extractDir = Join-Path $downloadDir "node-$version-extract"
        New-Item -ItemType Directory -Force $downloadDir | Out-Null
        Write-Host "[NODE] Downloading portable Node.js $version..." -ForegroundColor Cyan
        Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/$archiveName" -OutFile $archivePath
        Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/SHASUMS256.txt" -OutFile $sumsPath
        $line = Get-Content $sumsPath | Where-Object { $_ -match ("\s" + [regex]::Escape($archiveName) + "\s*$") } | Select-Object -First 1
        if (!$line) { throw "Node.js checksum entry not found for $archiveName" }
        $expected = ($line -split '\s+')[0].ToLowerInvariant()
        $actual = (Get-Sha256 $archivePath).ToLowerInvariant()
        if ($actual -ne $expected) { throw "Node.js SHA-256 mismatch for $archiveName" }
        if (Test-Path $extractDir) { Remove-Item -LiteralPath $extractDir -Recurse -Force }
        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDir -Force
        $unpacked = Join-Path $extractDir "node-v$version-win-x64"
        New-Item -ItemType Directory -Force $nodeDir | Out-Null
        Get-ChildItem -LiteralPath $unpacked -Force | Move-Item -Destination $nodeDir -Force
        Remove-Item -LiteralPath $extractDir -Recurse -Force
        return $nodeDir
    }
    # A moved venv has absolute interpreter paths. Preserve it instead of reusing it.
    if (Test-Path (Join-Path $stateDir 'venv')) {
        $target = [IO.Path]::GetFullPath((Join-Path $stateDir 'venv'))
        if (!$target.StartsWith([IO.Path]::GetFullPath($stateDir) + '\')) { throw 'Invalid venv path' }
        Move-Item -LiteralPath $target -Destination ($target + '.old-' + [DateTime]::Now.ToString('yyyyMMddHHmmss'))
    }
    if (Test-Path $stateFile) { Remove-Item -LiteralPath $stateFile }
    Run-Uv @('python','install','3.11.9')
    Run-Uv @('venv','--python','3.11.9', (Join-Path $stateDir 'venv'))
    $index = if ($Backend -eq 'cuda') { 'cu128' } else { 'cpu' }
    Run-Uv @('pip','install','--python',$python,'torch==2.10.0','torchaudio==2.10.0','torchvision==0.25.0','--index-url',"https://download.pytorch.org/whl/$index")
    Run-Uv @('pip','install','--python',$python,'-r',(Join-Path $PSScriptRoot 'requirements-app.txt'))
    # descript-audiotools pins protobuf below 3.20, while modern ONNX needs
    # protobuf 3.20.2+; install the audio stack first, then the ONNX stack
    # without asking the resolver to reconcile those incompatible metadata.
    Run-Uv @('pip','install','--python',$python,'--upgrade','protobuf>=4.25.1,<6','ml_dtypes>=0.5.4','flatbuffers','coloredlogs','packaging','sympy')
    Run-Uv @('pip','install','--python',$python,'--no-deps','onnx>=1.16,<2','onnxruntime>=1.24,<2','onnxscript>=0.2','onnx_ir>=0.1')
    if (Test-Path -LiteralPath $prebuiltEditor -PathType Leaf) {
        Write-Host '[EDITOR] Using the bundled Electron editor.' -ForegroundColor Cyan
    } else {
        $nodeDir = Ensure-PortableNode
        $nodeExe = Join-Path $nodeDir 'node.exe'
        # Invoke the JS entry point directly so cmd.exe does not reparse the
        # portable Node path (which can contain spaces or shell metacharacters).
        $npxCli = Join-Path $nodeDir 'node_modules\npm\bin\npx-cli.js'
        $env:Path = "$nodeDir;$env:Path"
        Write-Host '[EDITOR] Installing frontend dependencies...' -ForegroundColor Cyan
        Push-Location $editorDir
        try {
            # --force recreates pnpm links copied incompletely from another PC/ZIP.
            & $nodeExe $npxCli --yes 'pnpm@10.28.2' install --frozen-lockfile --force
            if ($LASTEXITCODE -ne 0) { throw "pnpm install failed ($LASTEXITCODE)" }
            if (!(Test-FrontendInstall)) {
                throw "Frontend install is incomplete: missing $frontendAsset"
            }
        } finally { Pop-Location }
    }
    if ($Backend -eq 'radeon') {
        $radeonVenv = Join-Path $box 'work\amd-dml-venv'
        Run-Uv @('venv','--python','3.11.9',$radeonVenv)
        $radeonPython = Join-Path $radeonVenv 'Scripts\python.exe'
        Run-Uv @('pip','install','--python',$radeonPython,'torch-directml','onnxruntime-directml','numpy','safetensors','einops')
    }
    # Install a pinned DACVAE revision: this package comes from a git archive
    # rather than a release, so a moving branch would silently change the
    # exported codec. Bump the commit hash deliberately.
    Run-Uv @('pip','install','--python',$python,'--no-deps','dacvae @ https://github.com/facebookresearch/dacvae/archive/414c20785fc3a28373073ea8ef7a1316eeeaca6e.zip')
    Write-Host '[MODEL] Downloading Irodori checkpoint from Hugging Face into models\ ...' -ForegroundColor Cyan
    & $python (Join-Path $PSScriptRoot 'prepare.py') --backend $Backend
    if ($LASTEXITCODE -ne 0) { throw 'Model preparation / runtime check failed. See logs/first-setup.log.' }
    if ($Backend -eq 'radeon') {
        Write-Host '[RADEON] Exporting the model-compatible DACVAE decoder to ONNX...' -ForegroundColor Cyan
        & $python (Join-Path $PSScriptRoot 'export_radeon_codec.py')
        if ($LASTEXITCODE -ne 0) { throw 'Radeon ONNX codec export failed. See logs/first-setup.log.' }
    }
    Write-Host '[LICENSES] Collecting installed runtime notices...'
    $licenseArgs = @((Join-Path $PSScriptRoot 'generate_runtime_licenses.py'))
    if ($Backend -eq 'radeon') { $licenseArgs += @('--site-packages', (Join-Path $radeonVenv 'Lib\site-packages')) }
    & $python @licenseArgs
    if ($LASTEXITCODE -ne 0) { throw 'Runtime license collection failed.' }
    @{identity=$identity;backend=$Backend;gpu=$gpu;python=$python;completed=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 ($stateFile + '.tmp')
    Move-Item -LiteralPath ($stateFile + '.tmp') -Destination $stateFile -Force
    Write-Host 'Setup complete. Starting the editor.'
} catch {
    Write-Host "SETUP FAILED: $_" -ForegroundColor Red
    Write-Host "Log: $(Join-Path $box 'logs\first-setup.log')"
    exit 1
} finally {
    if ($lock) { $lock.Dispose() }
    try { Stop-Transcript | Out-Null } catch {}
}
