# Cornerstone local startup helper for Windows PowerShell.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

& .\.venv\Scripts\cornerstone.exe stack up --migrate
Write-Host ""
Write-Host "Starting API. Open another terminal for worker/proof commands."
& .\.venv\Scripts\cornerstone.exe api --reload
