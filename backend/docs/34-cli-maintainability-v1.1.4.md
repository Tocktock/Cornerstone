# v1.1.4 — CLI Maintainability Cleanup

## Purpose

`v1.1.4` keeps the backend product behavior unchanged and improves the CLI codebase so the pilot/operator surface can continue to grow without turning into one large file.

The backend MVP loop remains the same:

```text
Live PostgreSQL
→ live Notion page
→ Artifact
→ EvidenceFragment
→ evidence review
→ official Concept
→ grounded context response
→ evaluation result
→ grounded_context_task_success_rate
```

## What changed

### CLI package split

The old single-file CLI was split into a package:

```text
src/cornerstone/cli/
  __init__.py
  __main__.py
  main.py
  parser.py
  models.py
  support.py
  config.py
  completion.py
  proof.py
  commands/
    __init__.py
    runtime.py
    source.py
    evidence.py
    concept.py
    context.py
    evaluation.py
    config.py
    completion.py
```

This keeps command parsing, command handlers, proof helpers, output helpers, HTTP helpers, and local configuration isolated.

### CLI profile/config support

The CLI now supports a small local profile file:

```bash
cornerstone config get
cornerstone config set baseUrl http://localhost:8000
cornerstone config set defaultReviewer reviewer@example.com
cornerstone config set defaultReportsDir reports
cornerstone config unset defaultReviewer
cornerstone config path
```

The config is stored at:

```text
~/.cornerstone/config.json
```

You can override the config directory with:

```bash
CORNERSTONE_CONFIG_DIR=/tmp/cornerstone-cli cornerstone config get
```

Tokens and secrets must not be stored in CLI config.

### Shell completion

The CLI can print lightweight completion scripts:

```bash
cornerstone completion zsh
cornerstone completion bash
cornerstone completion powershell
```

### Output behavior

The existing command output modes are preserved:

```text
human-readable tables by default
--json for automation
clear next-action messages on API reachability errors
```

## Why this is not over-engineering

This cleanup does not add new backend semantics, connectors, AI behavior, or persistence features. It only separates responsibilities so future CLI additions are easier to review.

## Exit criteria

`v1.1.4` is complete when:

```text
1. The CLI is importable through cornerstone.cli.
2. The console script still resolves to cornerstone.cli:main.
3. Product-loop commands still work.
4. proof run still produces consolidated reports.
5. config and completion commands work.
6. package hygiene excludes cache/build/secret files.
7. release-candidate check passes for v1.1.4.
```

## Known limitations

```text
- The CLI remains a pilot/operator tool, not a full UI.
- Shell completion is lightweight and command-level, not full dynamic completion.
- CLI config stores only non-secret preferences.
- Live PostgreSQL and live Notion gates still require local environment setup.
```
