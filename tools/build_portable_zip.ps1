param(
    [string]$OutputPath = '',
    [switch]$SkipEditorBuild
)
# Builds the portable release: the Electron editor (editor\), the first-setup
# launcher (kataribe.exe from tools\Launcher.cs) and the engine bootstrap
# (editor\irodori-engine.exe from voicevox-editor\build\IrodoriEngineBootstrap.cs),
# plus the git-tracked runtime files. Python, packages and models are not
# bundled: kataribe.exe runs tools\setup.ps1 on first start to download them.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\build_portable_zip.ps1 [-SkipEditorBuild]
# Output: dist\kataribe-portable-<version>.zip (top folder: kataribe\).

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8  # git ls-files prints UTF-8 paths
$box = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$editorDir = Join-Path $box 'voicevox-editor'
$version = (Get-Content (Join-Path $editorDir 'package.json') -Raw | ConvertFrom-Json).version
if (!$OutputPath) { $OutputPath = Join-Path $box "dist\kataribe-portable-$version.zip" }
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
$stageRoot = Join-Path $box 'dist\stage'
$stage = Join-Path $stageRoot 'kataribe'
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$sevenZip = Join-Path $editorDir 'vendored\7z\7za.exe'
foreach ($tool in @($csc, $sevenZip)) {
    if (!(Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Required tool not found: $tool" }
}

# 1. Editor (electron-builder --dir -> voicevox-editor\dist_electron\win-unpacked)
$unpacked = Join-Path $editorDir 'dist_electron\win-unpacked'
if (!$SkipEditorBuild) {
    $nodeDir = Join-Path $box '.local\node\24.11.1'
    $node = Join-Path $nodeDir 'node.exe'
    if (!(Test-Path -LiteralPath $node)) { $node = (Get-Command node.exe -ErrorAction Stop).Source; $nodeDir = Split-Path $node }
    $npxCli = Join-Path $nodeDir 'node_modules\npm\bin\npx-cli.js'
    $env:Path = "$nodeDir;$env:Path"
    $env:CI = 'true'  # keeps pnpm from asking for a TTY confirmation
    Push-Location $editorDir
    try {
        if (!(Test-Path (Join-Path $editorDir 'node_modules'))) {
            Write-Host '[1/5] Installing editor dependencies...' -ForegroundColor Cyan
            & $node $npxCli --yes 'pnpm@10.28.2' install --frozen-lockfile
            if ($LASTEXITCODE -ne 0) { throw "pnpm install failed ($LASTEXITCODE)" }
        }
        Write-Host "[1/5] Building the editor $version (electron:build:dir)..." -ForegroundColor Cyan
        & $node $npxCli --yes 'pnpm@10.28.2' run electron:build:dir
        if ($LASTEXITCODE -ne 0) { throw "Editor build failed ($LASTEXITCODE)" }
    } finally { Pop-Location }
} else {
    Write-Host '[1/5] Using the existing editor build.' -ForegroundColor Cyan
}
if (!(Test-Path -LiteralPath (Join-Path $unpacked 'kataribe.exe'))) { throw "Editor build not found: $unpacked\kataribe.exe" }

# 2. Stage the tree
Write-Host "[2/5] Staging $stage ..." -ForegroundColor Cyan
if (Test-Path -LiteralPath $stageRoot) { Remove-Item -LiteralPath $stageRoot -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item -Path (Join-Path $unpacked '*') -Destination (New-Item -ItemType Directory -Force (Join-Path $stage 'editor')) -Recurse -Force
Copy-Item -LiteralPath (Join-Path $editorDir 'build\irodori-engine-path.txt') -Destination (Join-Path $stage 'editor')
Copy-Item -LiteralPath (Join-Path $editorDir 'IRODORI_EDITOR.md') -Destination (Join-Path $stage 'editor')

# Runtime files come from git so that new modules are picked up automatically and
# local experiments, caches and tests stay out. Development-only files are skipped.
$include = '^(LICENSE|LICENSES\.md|setup\.bat|bat/|irodori-tts/|runtime/|licenses/|models/|speakers/|tools/|voicevox-editor/public/[^/]*licenses\.json$)'
$exclude = '^(irodori-tts/tests/|tools/(Launcher\.cs|build_portable_zip\.ps1|build_english_dictionary\.py|bench_[^/]*|test_[^/]*|setup_colab\.sh|release/)|models/\.gitattributes$)'
$files = @(& git -C $box -c core.quotepath=false ls-files) | Where-Object { $_ -match $include -and $_ -notmatch $exclude }
if ($LASTEXITCODE -ne 0 -or !$files.Count) { throw 'git ls-files failed.' }
foreach ($file in $files) {
    $target = Join-Path $stage $file
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Copy-Item -LiteralPath (Join-Path $box $file) -Destination $target
}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'release') -File) {
    Copy-Item -LiteralPath $file.FullName -Destination $stage
}
Write-Host "      $($files.Count) tracked files"

# 3. Launcher and engine bootstrap (.NET Framework 4, ships with Windows 10 / 11)
Write-Host '[3/5] Compiling kataribe.exe and irodori-engine.exe...' -ForegroundColor Cyan
Add-Type -AssemblyName System.Drawing
$icon = Join-Path $stageRoot 'kataribe.ico'
$stream = [IO.File]::Create($icon)
try { [Drawing.Icon]::ExtractAssociatedIcon((Join-Path $unpacked 'kataribe.exe')).Save($stream) } finally { $stream.Dispose() }
& $csc /nologo /codepage:65001 /optimize+ /target:winexe /platform:anycpu "/win32icon:$icon" /reference:System.Windows.Forms.dll "/out:$(Join-Path $stage 'kataribe.exe')" (Join-Path $PSScriptRoot 'Launcher.cs')
if ($LASTEXITCODE -ne 0) { throw 'Launcher compile failed.' }
& $csc /nologo /codepage:65001 /optimize+ /target:exe /platform:anycpu "/out:$(Join-Path $stage 'editor\irodori-engine.exe')" (Join-Path $editorDir 'build\IrodoriEngineBootstrap.cs')
if ($LASTEXITCODE -ne 0) { throw 'Engine bootstrap compile failed.' }
Remove-Item -LiteralPath $icon

# 4. Sanity checks on the staged layout
Write-Host '[4/5] Checking the layout...' -ForegroundColor Cyan
foreach ($required in @('kataribe.exe','START_HERE.bat','setup.bat','tools\setup.ps1','tools\prepare.py',
                         'editor\kataribe.exe','editor\irodori-engine.exe','editor\irodori-engine-path.txt',
                         'irodori-tts\wrapper\editor_engine.py','runtime\trt-lab\repo\irodori_tts\model.py',
                         'speakers\tsukuyomi\tsukuyomi.speaker.safetensors')) {
    if (!(Test-Path -LiteralPath (Join-Path $stage $required))) { throw "Missing from the package: $required" }
}

# 5. Zip (7-Zip keeps the Japanese file names as UTF-8)
Write-Host "[5/5] Writing $OutputPath ..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force (Split-Path $OutputPath) | Out-Null
if (Test-Path -LiteralPath $OutputPath) { Remove-Item -LiteralPath $OutputPath }
Push-Location $stageRoot
try {
    & $sevenZip a -tzip -mx=7 -mcu=on $OutputPath 'kataribe' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed ($LASTEXITCODE)" }
} finally { Pop-Location }
$size = (Get-Item -LiteralPath $OutputPath).Length / 1MB
Write-Output ("Portable package: {0} ({1:N0} MB)" -f $OutputPath, $size)
