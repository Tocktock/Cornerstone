# Cornerstone Agent Guide

## Current State

Cornerstone is intentionally reset to an empty product baseline.

Do not infer current product architecture, runtime behavior, data model, or
technical direction from the removed retrieval-augmented support workspace. That
product surface is no longer the active baseline.

## Operating Rules

- Follow the higher-priority global instructions in the active Codex context.
- Work directly in this repository; do not use git worktrees.
- Treat untracked local files and archives as user-owned unless the user
  explicitly approves deleting them.
- Prefer small, reversible changes until the new product direction is defined.
- Verify claims with command output, file paths, or explicit assumptions.

## Rebuild Guidance

- Start from the new product brief and acceptance criteria.
- Add source code, dependencies, runtime configuration, and tests only after the
  new direction is explicit.
- When a stack is chosen, update this file with the real runbook and quality
  gates for that stack.

## Current Verification Surface

No application or automated product test suite exists in this reset baseline.
Until a new stack lands, use repository-level checks such as `git status`,
`git diff --check`, and targeted file inspection.
