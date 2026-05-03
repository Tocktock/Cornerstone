# v1.1.1 — CLI Product-Loop Operations

## Goal

Make the proven Cornerstone backend MVP loop operable without curl.

The backend v1.0 path already proved:

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

v1.1.1 adds workflow-oriented CLI commands for the same loop.

## Why this matters

The next product risk is not backend capability. It is pilot usability.

A Source Admin, reviewer, or product owner should be able to inspect source state, review evidence, officialize Concepts, ask grounded questions, and run evaluation tasks without remembering API paths or JSON payload shapes.

## Added CLI workflows

### Source Studio

```bash
cornerstone source list
cornerstone source show <source-id>
cornerstone source objects <source-id>
cornerstone source jobs <source-id>
cornerstone source sync <source-id>
```

These commands expose source state, connection state, freshness, object discovery, sync jobs, and next actions.

### Evidence review

```bash
cornerstone evidence queue
cornerstone evidence show <evidence-id>
cornerstone evidence review <evidence-id> --reviewer reviewer@example.com
cornerstone evidence reject <evidence-id> --reviewer reviewer@example.com --note "Not accurate"
cornerstone evidence conflict <evidence-id> --reviewer reviewer@example.com --note "Conflicts with another source"
```

These commands make the human review step explicit. Extracted evidence is not official until reviewed.

### Concept officialization

```bash
cornerstone concept create-from-evidence <evidence-id> \
  --name "Cornerstone" \
  --definition "Cornerstone is a shared organizational context layer." \
  --created-by reviewer@example.com

cornerstone concept officialize <concept-id> --reviewer reviewer@example.com
cornerstone concept list --status official
cornerstone concept show <concept-id>
```

These commands preserve the product rule: official context must be reviewed and evidence-backed.

### Grounded answers

```bash
cornerstone ask "What is Cornerstone?"
cornerstone context query "What is Cornerstone?" --json
```

Human-readable output includes answer, trust label, freshness, citations, and limitations.

### Evaluation

```bash
cornerstone eval create \
  --name "Cornerstone definition" \
  --query "What is Cornerstone?" \
  --expected-trust-label official \
  --expected-answer-contains "shared organizational context layer" \
  --required-evidence <evidence-id> \
  --required-concept <concept-id> \
  --require-official-answer \
  --created-by reviewer@example.com

cornerstone eval run <task-id>
cornerstone eval results
cornerstone eval summary
```

Evaluation keeps the product aligned with `grounded_context_task_success_rate` instead of superficial answer generation.

### Proof runner dry-run

```bash
cornerstone proof run --postgres --notion --product-loop --dry-run
```

This creates an ordered proof plan that can later be converted into a full proof runner.

## Output modes

Read commands support human-readable output by default and `--json` for automation:

```bash
cornerstone source list --json
cornerstone evidence queue --json
cornerstone ask "What is Cornerstone?" --json
cornerstone eval summary --json
```

## Safety and next actions

The CLI prints next actions for common workflow points. For example, a queued sync job suggests running the worker, and Concept creation suggests officialization after review.

The CLI does not store Notion tokens. Live Notion credentials remain environment-only.

## Non-goals

```text
No full terminal UI.
No CLI plugin system.
No new connector semantics.
No frontend replacement.
No OAuth browser automation.
```

## Test coverage

v1.1.1 adds CLI unit coverage for:

```text
source list formatting
evidence review payloads
Concept candidate creation payloads
grounded answer formatting
evaluation task payloads
proof dry-run report generation
```

## Exit criteria

v1.1.1 is successful when a pilot operator can complete the backend MVP loop mostly through `cornerstone` commands instead of curl.
