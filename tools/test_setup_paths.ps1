# Run with Windows PowerShell 5.1: powershell.exe -NoProfile -File tools/test_setup_paths.ps1
$ErrorActionPreference = 'Stop'
$sourceRoot = Split-Path $PSScriptRoot -Parent
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('irodori setup ' + [guid]::NewGuid().ToString('N'))
$box = Join-Path $testRoot 'project with spaces (copy) & test!'
New-Item -ItemType Directory -Path (Join-Path $box 'bat'), (Join-Path $box 'tools') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $sourceRoot 'setup.bat') -Destination $box
Copy-Item -LiteralPath (Join-Path $sourceRoot 'bat\first_setup.bat') -Destination (Join-Path $box 'bat')
$setup = Join-Path $box 'tools\setup.ps1'
# Replace only the expensive installer; exercise both real batch entry points.
@'
param([string]$Backend, [switch]$Check, [switch]$Force)
if ($Backend -ne 'cpu' -or !$Check) { exit 91 }
if ((Get-Location).Path -ne (Split-Path $PSScriptRoot -Parent)) { exit 92 }
Write-Output 'REACHED_SETUP'
if ($Force) { exit 17 }
exit 0
'@ | Set-Content -LiteralPath $setup -Encoding UTF8
foreach ($entry in @('setup.bat', 'bat\first_setup.bat')) {
    foreach ($expected in @(0, 17)) {
        $forceArg = if ($expected -eq 17) { ' -Force' } else { '' }
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $env:ComSpec
        $info.Arguments = '/d /s /c ""' + (Join-Path $box $entry) + '" -Backend cpu -Check' + $forceArg + ' <nul"'
        $info.WorkingDirectory = $testRoot
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = [Diagnostics.Process]::Start($info)
        $output = $process.StandardOutput.ReadToEnd()
        $errors = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if ($process.ExitCode -ne $expected) { throw "$entry returned $($process.ExitCode), expected $expected`: $output $errors" }
        # first_setup.bat no longer redirects the installer into a log file
        # (setup.ps1 writes logs\first-setup.log itself), so check its output.
        if ($output -notmatch 'REACHED_SETUP') { throw 'Installer was not reached' }
        Write-Host "PASS $entry exit=$expected"
        $process.Dispose()
    }
}
# Exercise the actual Start-Process expression used by environment repair.
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $sourceRoot 'tools\rebuild_environment.ps1'), [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw $parseErrors[0] }
$start = $ast.Find({ param($node) $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'Start-Process' }, $true)
'param([string]$Backend); if ($Backend -eq "cpu") { exit 17 }; exit 91' | Set-Content -LiteralPath $setup -Encoding UTF8
$Backend = 'cpu'
$process = & ([scriptblock]::Create($start.Extent.Text))
if ($process.ExitCode -ne 17) { throw "Repair invocation returned $($process.ExitCode), expected 17" }
$process.Dispose()
Write-Host 'PASS repair invocation with spaces and exit-code propagation'
Write-Host "Fixtures: $testRoot"
