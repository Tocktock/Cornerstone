# ADR-0008: ConnectorHub Review Split and Module Map

## Status

Accepted as review strategy for PR #20 while it remains draft.

## Context

PR #20 is a broad ConnectorHub adoption overlay. It combines scenario contract cleanup, compact evidence, CLI/runtime adapter behavior, human-gate preparation, and local VS2 rehearsal references. Reviewer feedback correctly identified two risks:

- the PR is too large to merge safely as one product change;
- `packages/cornerstone_cli/connector.py` and `tests/scenario/test_connectorhub_cli.py` are too broad to remain the long-term ownership shape.

This ADR does not claim that the split has already happened. It records the required review and migration map so the current PR can stay a local evidence branch while future PRs are cut into safer review slices.

## Decision

Treat PR #20 as a draft evidence branch. Do not present it as a ConnectorHub milestone-complete merge candidate. The preferred split sequence is:

1. Contract and matrix cleanup only.
2. Compact evidence layout and verifier support.
3. ConnectorPort setup and delivery core for `CS-CH-001` through `CS-CH-014`.
4. GitHub selected-repository read-only lane for `CS-CH-015` through `CS-CH-020`.
5. macOS, Chrome, Watch Rule, and Watch Result local fixture lane for `CS-CH-021` through `CS-CH-028`.
6. Local fixture Action lane for `CS-CH-029` through `CS-CH-033`.
7. Scope, credential custody, egress, audit bridge, and product-surface guards for `CS-CH-034` through `CS-CH-040`.
8. VS2 production-like local rehearsal as a separate local/manual proof slice.

The target module map for ConnectorHub runtime code is:

```text
packages/cornerstone_cli/connector/
  __init__.py
  contracts.py
  setup.py
  delivery.py
  evidence.py
  raw_access.py
  github_readonly.py
  capture_macos.py
  capture_chrome.py
  watch_rules.py
  watch_results.py
  action_preflight.py
  audit_bridge.py
  human_gates.py
  validation.py
```

The matching test split should keep scenario-first coverage but move broad fixtures and assertions out of one monolithic file:

```text
tests/scenario/connectorhub/
  test_setup_delivery.py
  test_evidence_raw_access.py
  test_github_readonly.py
  test_capture_watch.py
  test_action_lane.py
  test_scope_credential_egress_audit.py
  test_human_gates.py
  test_compact_reports.py
```

## Boundaries

- This ADR is a module map and split plan, not evidence that the refactor has already happened.
- No production readiness, live-provider readiness, or human acceptance is implied.
- No GitHub Actions workflow is required for this ADR; reviewers should use `scripts/verify_connectorhub_local_evidence.sh`.
- The current PR should remain draft until the owner chooses either to split it or to merge a deliberately scoped local-evidence slice.

## Consequences

Positive:

- Review and rollback can happen by domain slice instead of one oversized merge.
- Future module ownership aligns with ConnectorHub domain boundaries.
- Generated evidence remains reviewable through local scripts while GitHub workflow changes are out of scope.

Negative:

- The current branch still contains the monolithic files until the split PRs are cut.
- The split will require careful compatibility work to preserve CLI schemas, scenario IDs, evidence refs, and report hashes.

## Verification

This ADR is guarded by:

- `scripts/verify_connectorhub_engineering_trail.py`
- `scripts/verify_connectorhub_local_evidence.sh`
- `docs/verification-reports/CONNECTOR_HUB_REVIEWER_GUIDE.md`
