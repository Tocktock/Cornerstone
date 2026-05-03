# Cross-platform Local Quickstart

This guide is the OS-agnostic starter for Cornerstone Backend.

## 1. Prerequisites

```text
Python 3.13+
Docker Desktop or Docker Engine with Docker Compose plugin
Git
```

For live Notion proof, you also need:

```text
Notion integration token
A Notion page shared with that integration
```

Do not commit tokens or `.env`.

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## 3. Safe setup

```bash
cornerstone setup --fix
cornerstone doctor
```

Windows:

```powershell
cornerstone setup windows --fix
cornerstone doctor
```

## 4. Start local services

```bash
cornerstone stack up --migrate
cornerstone api --reload
```

Or use OS scripts:

```text
macOS/Linux: scripts/start_local.sh
Windows:     scripts\windows_start_local.ps1
```

## 5. Run proof

```bash
cornerstone proof run --all --continue-on-failure --markdown --save reports/cornerstone-live-proof.json
```

## 6. Reset local database

This deletes the local Docker database volume.

```bash
cornerstone local reset --yes --start-after --migrate
```
