# vX.Y.Z — <Feature or Release Name>

## Purpose

Explain why this version exists and what product or engineering problem it solves.

## Version goal

State the single measurable goal of this version in one or two sentences.

A good version goal has this shape:

```text
Given <starting state>, Cornerstone must enable <new capability>, while preserving <trust boundary>.
```

## Confirmed non-goal

State what this version must not do.

Examples:

```text
- no automatic officialization
- no graph depth above 1
- no live external LLM provider
- no frontend UI
- no database migration
```

## User value

Describe what the user can do after this version that they could not do before.

## Scope

Included:

```text
- item
- item
```

Deferred:

```text
- item
- item
```

## Product behavior

Describe expected behavior in user-facing terms.

## Trust and safety rules

```text
- rule
- rule
```

## API contract

List new or changed endpoints.

```text
GET /v1/example
POST /v1/example
```

If the version is documentation-only, say so clearly.

## Data model changes

```text
- ModelName.fieldName
- new_table_name
```

If there are no data model changes, say so clearly.

## Measurable acceptance checklist

Every checklist row must be measurable. Avoid vague rows such as "update docs" unless the exact document and expected content are named.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| VX-01 | The new endpoint returns the documented response shape. | Integration test name, OpenAPI snapshot, or verification report. | pending |
| VX-02 | The trust boundary is preserved. | Negative test, service assertion, or release note section. | pending |
| VX-03 | The version document states goal and non-goal. | This document. | pending |

## Implementation checklist

```text
[ ] Task
[ ] Task
```

Implementation checklist items track engineering work. The measurable acceptance checklist above defines whether the version is releasable.

## Test plan

```text
[ ] Unit tests
[ ] Integration tests
[ ] Contract tests
[ ] Regression tests
```

## Proof / verification

```bash
cornerstone version
python scripts/check_release_candidate.py
```

## Chronicle position

State the previous version and next handoff.

```text
Previous: vX.Y.(Z-1) — <name>
This version: vX.Y.Z — <name>
Next: vX.Y.(Z+1) — <name>
```

## Known limitations

```text
- limitation
- limitation
```

## Exit criteria

```text
[ ] Version metadata updated.
[ ] Docs updated.
[ ] Measurable acceptance checklist completed.
[ ] Tests pass.
[ ] Release checker passes.
[ ] Next-version handoff documented.
```

## Next version handoff

Describe what the next release should implement.
