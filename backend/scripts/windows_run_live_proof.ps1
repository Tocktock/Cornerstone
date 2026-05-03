# Cornerstone live proof helper for Windows PowerShell.
# Requires RUN_POSTGRES_TESTS=1 and, for Notion, RUN_NOTION_E2E=1 plus Notion env vars.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

& .\.venv\Scripts\cornerstone.exe proof run --all --continue-on-failure --markdown --save reports/cornerstone-live-proof.json
