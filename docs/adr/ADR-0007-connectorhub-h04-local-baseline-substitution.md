# ADR-0007: ConnectorHub H04 Local Baseline Substitution

## Status

Accepted for local review input only.

## Context

ConnectorHub human gate `CS-CH-H04` asks for production-like RequestContext, PostgreSQL/RLS, OPA, egress, backup/restore, and audit readiness review. The PR evidence branch can generate local VS2 and ConnectorHub reports, but it cannot collect the dated human production-like acceptance record itself.

The reviewer feedback approved replacing unavailable manual checks with local integration evidence where automation can verify the behavior, provided the result remains clearly outside human acceptance and production readiness.

## Decision

Use current local VS2 and ConnectorHub dependency reports as H04 baseline review inputs. The local baseline may include status summaries, hashes, recommended preflight commands, and required human-delta descriptions. It must preserve:

- `acceptance_sufficient=false`
- `product_claim_allowed=false`
- `pass_claim_allowed=false`
- `claim_boundary=h04_local_baseline_snapshot_is_review_input_not_human_acceptance`

The local baseline does not close `CS-CH-H04`. A dated human record is still required before H04 can move from `HUMAN_REQUIRED` to `PASS`.

## Consequences

Positive:

- Reviewers can see concrete local evidence before attempting production-like validation.
- The local evidence can be regenerated and hashed in the engineering trail.
- The H04 handoff is less ambiguous because the remaining human delta is explicit.

Negative:

- Local topology and synthetic fixtures can still miss production network, IdP, secret-backend, migration, and recovery risks.
- The PR must keep draft/needs-follow-up status until human evidence exists or the merge strategy deliberately lands only the local adoption overlay.

## Verification

The boundary is guarded by:

- `reports/scenario/connectorhub-human-gate-preflight-bundle-cs-ch-h04-2026-06-24.json`
- `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json`
- `reports/scenario/connectorhub-human-gate-next-2026-06-24.json`
- `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json`
- `python3 scripts/verify_connectorhub_engineering_trail.py`
