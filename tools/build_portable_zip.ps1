param(
    [string]$OutputPath = (Join-Path $PSScriptRoot '..\dist\kataribe-portable.zip')
)

$ErrorActionPreference = 'Stop'
$box = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$extras = Join-Path $box 'portable'
$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory -and !(Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

# The portable release is the same tree as the installer payload plus the
# launcher (START_HERE.bat) and the guide, so the payload builder does the work.
# It runs setup.ps1 on first start and never touches the registry, so an
# extracted copy is a complete, self-contained installation.
& (Join-Path $PSScriptRoot 'build_installer_payload.ps1') -OutputPath $OutputPath -Extras $extras

Write-Output "Portable package: $OutputPath"
