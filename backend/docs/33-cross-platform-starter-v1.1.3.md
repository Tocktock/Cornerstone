# v1.1.3 — Cross-platform Local Starter

## Purpose

This guide supports the Cornerstone backend local operator workflow.

`v1.1.3` turns the macOS-only starter idea into an OS-agnostic local operator kit for:

```text
macOS
Linux
Windows PowerShell
```

The goal is not to add backend semantics. The goal is to make the proven backend MVP loop easier to operate locally and during pilots.

## New CLI Commands

```bash
cornerstone setup
cornerstone setup --fix
cornerstone setup windows --json
cornerstone doctor --fix
cornerstone local reset --yes --start-after --migrate
```

## Scripts

Unix-like systems:

```bash
scripts/setup_local.sh
scripts/start_local.sh
scripts/run_live_proof.sh
```

Windows PowerShell:

```powershell
scripts\windows_setup.ps1
scripts\windows_start_local.ps1
scripts\windows_run_live_proof.ps1
```

Compatibility wrappers remain:

```bash
scripts/macos_setup.sh
scripts/macos_start_local.sh
scripts/macos_run_live_proof.sh
scripts/linux_setup.sh
scripts/linux_start_local.sh
scripts/linux_run_live_proof.sh
```

## Safety Rules

`cornerstone doctor --fix` and `cornerstone setup --fix` are intentionally safe. They may:

```text
create .env if missing
create reports/
chmod shell scripts on Unix-like systems
```

They do not overwrite secrets and do not delete local data.

`cornerstone local reset` is destructive and requires:

```bash
cornerstone local reset --yes
```

## Exit Criteria

`v1.1.3` succeeds when a new operator on macOS, Linux, or Windows can:

```text
1. Run setup.
2. Start local PostgreSQL.
3. Run migrations.
4. Start the API.
5. Run the one-command proof runner.
```

The CLI remains a thin workflow layer over the backend API and scripts.
