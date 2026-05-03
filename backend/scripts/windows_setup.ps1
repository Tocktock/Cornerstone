# Cornerstone local setup helper for Windows PowerShell.
# Run from the repository root or from scripts/.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.13+ is required. Install Python and rerun."
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m venv .venv
} else {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -e ".[dev]"
& .\.venv\Scripts\cornerstone.exe doctor --fix

Write-Host ""
Write-Host "Local setup complete."
Write-Host "Next:"
Write-Host "  .\.venv\Scripts\cornerstone.exe stack up --migrate"
Write-Host "  .\.venv\Scripts\cornerstone.exe api --reload"
