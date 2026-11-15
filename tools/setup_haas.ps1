# Haas-ja ASR environment for speaker training (transcribes audio that has no text).
# Creates .local\venv-haas (NeMo 2.7.3 + PyTorch) next to the app venv, so the app venv stays small.
# The Haas inference code is bundled in irodori-tts\wrapper\vendor\haas_asr and the Triton kernels are
# replaced with irodori-tts\wrapper\haas_kda.py, so this works on Windows with CUDA or CPU.
# Keep this file ASCII only (Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI).
param(
  [switch]$Force,
  [ValidateSet('auto', 'cuda', 'cpu')][string]$Device = 'auto',
  [switch]$SkipModel
)
$ErrorActionPreference = 'Stop'
$box = Split-Path $PSScriptRoot -Parent
$stateDir = Join-Path $box '.local'
$venv = Join-Path $stateDir 'venv-haas'
$python = Join-Path $venv 'Scripts\python.exe'
$uv = Join-Path $stateDir 'bin\uv.exe'
$wrapper = Join-Path $box 'irodori-tts\wrapper'
$requirements = Join-Path $PSScriptRoot 'requirements-haas.txt'
New-Item -ItemType Directory -Force (Join-Path $box 'logs') | Out-Null
Start-Transcript -Path (Join-Path $box 'logs\haas-setup.log') -Append | Out-Null
try {
  if (!(Test-Path -LiteralPath $uv -PathType Leaf)) { throw 'Project-local uv is missing. Run setup.bat first.' }
  $env:UV_PYTHON_INSTALL_DIR = Join-Path $stateDir 'python'
  $env:UV_CACHE_DIR = Join-Path $stateDir 'uv-cache'
  if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $box '.cache\huggingface' }

  if ($Device -eq 'auto') {
    $Device = 'cpu'
    $state = Join-Path $stateDir 'setup.json'
    if (Test-Path $state) {
      try { if ((Get-Content $state -Raw | ConvertFrom-Json).backend -eq 'cuda') { $Device = 'cuda' } } catch {}
    } elseif (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
      $Device = 'cuda'
    }
  }
  $index = if ($Device -eq 'cuda') { 'cu128' } else { 'cpu' }
  Write-Host "[1/5] Haas-ja environment: $venv (PyTorch $index)" -ForegroundColor Cyan

  if ($Force -and (Test-Path $venv)) { Remove-Item -Recurse -Force $venv }
  if (!(Test-Path $python)) {
    & $uv venv $venv --python 3.11
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed ($LASTEXITCODE)." }
  }

  Write-Host '[2/5] Installing PyTorch (progress is shown)...' -ForegroundColor Cyan
  & $uv pip install --python $python 'torch==2.10.0' 'torchaudio==2.10.0' --index-url "https://download.pytorch.org/whl/$index"
  if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed ($LASTEXITCODE)." }

  Write-Host '[3/5] Installing NeMo and the other pinned packages...' -ForegroundColor Cyan
  # The list is the complete frozen set of a tested environment (some pins are outside the ranges that
  # NeMo's dependencies declare), so install it as-is without resolving again.
  & $uv pip install --python $python --no-deps -r $requirements
  if ($LASTEXITCODE -ne 0) { throw "Package install failed ($LASTEXITCODE)." }

  Write-Host '[4/5] Checking that Haas-ja loads...' -ForegroundColor Cyan
  $check = "import sys; sys.path[:0] = [r'$wrapper', r'$wrapper\vendor']; import haas_kda; haas_kda.install(); import haas_asr, torch; print('haas_asr OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  & $python -c $check
  if ($LASTEXITCODE -ne 0) { throw 'Haas-ja import check failed.' }

  if (!$SkipModel) {
    Write-Host '[5/5] Downloading the Haas-ja-v1.0 model (about 1.2 GB, first time only)...' -ForegroundColor Cyan
    $load = "import sys; sys.path[:0] = [r'$wrapper', r'$wrapper\vendor']; import haas_kda; haas_kda.install(); import haas_asr; haas_asr.load_model('third-intelligence/Haas-ja-v1.0', device='cpu'); print('model OK')"
    & $python -c $load
    if ($LASTEXITCODE -ne 0) { throw 'Model download or load failed.' }
  } else {
    Write-Host '[5/5] Skipped the model download (it is downloaded on first use).' -ForegroundColor Yellow
  }
  Write-Host "Haas-ja setup complete: $python" -ForegroundColor Green
} catch {
  Write-Host "HAAS SETUP FAILED: $_" -ForegroundColor Red
  exit 1
} finally { try { Stop-Transcript | Out-Null } catch {} }
