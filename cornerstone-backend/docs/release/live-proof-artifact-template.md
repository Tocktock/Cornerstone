# Live Proof Artifact Template

Use this template for each backend release candidate live proof.

## Metadata

```text
Version:
Date:
Operator:
Environment:
Database name:
Notion page description:
```

## Local package gate

```text
Command:
Result:
pytest:
coverage:
ruff:
mypy:
compileall:
package hygiene:
```

## Live PostgreSQL gate

```text
Command:
passed:
skipped:
failed:
errors:
extensions verified:
notes:
```

## Live Notion gate

```text
Command:
status:
sync_job_status:
artifact_count:
evidence_fragment_count:
source_next_action:
notes:
```

## API product-loop proof

```text
healthz version:
hasRealSources:
source type:
source authStatus:
source connectionStatus:
artifact count:
evidence count:
reviewed evidence count:
official concept count:
grounded trustLabel:
officialAnswerAvailable:
invalidCitationCount:
evaluationSuccess:
groundedContextTaskSuccessRate:
```

## Safety negatives

```text
unauthorized review:
unauthorized officialization:
fake provider source creation:
fake OAuth completion:
legacy source sync:
manual sync on Notion source:
weak evaluation task:
```

## Release decision

```text
status: passed | blocked
blockers:
accepted limitations:
next action:
```

