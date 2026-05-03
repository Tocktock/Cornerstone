# Windows PowerShell Quickstart

This guide runs the Cornerstone backend locally.

## Prerequisites

```text
Python 3.13+
Docker Desktop for Windows
Git for Windows
PowerShell
```

## Setup

```powershell
.\scripts\windows_setup.ps1
```

Or manually:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cornerstone setup windows --fix
```

## Start

```powershell
.\scripts\windows_start_local.ps1
```

## Live proof

Set environment variables in PowerShell:

```powershell
$env:RUN_POSTGRES_TESTS = "1"
$env:RUN_NOTION_E2E = "1"
$env:NOTION_MOCK_EXTERNAL_API = "false"
$env:CONNECTOR_ENCRYPTION_SECRET = "replace-with-a-long-local-proof-secret-32chars-plus"
$env:NOTION_E2E_ACCESS_TOKEN = "<your-token>"
$env:NOTION_E2E_PAGE_ID = "<your-page-id>"
```

Then run:

```powershell
.\scripts\windows_run_live_proof.ps1
```

Do not write the Notion token to `.env` or commit it.
