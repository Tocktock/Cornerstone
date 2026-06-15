# CornerStone Reference Images

**Status:** Active design reference set for future UI work.
**Owner:** JiYong / Tars.
**Date added:** 2026-06-15.

## Purpose

This directory stores the current visual reference images for CornerStone UI direction.

These images are design references only. They do not implement product behavior, do not mark any scenario as `PASS`, do not prove VS0 human operator acceptance, and do not imply production release readiness, live-provider readiness, or autonomous external writeback.

Future UI implementation must still follow:

1. `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`
2. `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md`
3. `docs/design/tokens/cornerstone_design_tokens_v0_3.json`
4. The task-specific scenario contract and verification report

If a reference image conflicts with product SoT, scenario contracts, safety rules, CLI-native-first gates, or evidence/audit/action boundaries, the product and safety rules win.

## Reference Set

| File | Concept surface | Use |
|---|---|---|
| [`cornerstone-reference-01-vendor-detail.png`](cornerstone-reference-01-vendor-detail.png) | Vendor object detail | Entity/object explorer, overview tabs, related artifacts, claims, connections, activity, key facts, risk, and trust. |
| [`cornerstone-reference-02-operations-inbox.png`](cornerstone-reference-02-operations-inbox.png) | Operations inbox | Needs-review triage, approval requests, policy blocked items, failed runs, and right-side item detail. |
| [`cornerstone-reference-03-admin-connectors.png`](cornerstone-reference-03-admin-connectors.png) | Admin connectors | Connector sources, policies, access roles, namespace settings, recent activity, and admin-only containment. |
| [`cornerstone-reference-04-search-results.png`](cornerstone-reference-04-search-results.png) | Search results | Universal search across artifacts, claims, entities, and actions with filters, trust chips, and suggested follow-ups. |
| [`cornerstone-reference-05-claim-draft-supporting-evidence.png`](cornerstone-reference-05-claim-draft-supporting-evidence.png) | Claim draft | Claim statement, rationale, trust ladder, supporting evidence picker, and review/promotion controls. |
| [`cornerstone-reference-06-artifact-viewer.png`](cornerstone-reference-06-artifact-viewer.png) | Artifact viewer | Original artifact preview, source metadata, summary, extracted entities, related claims, and provenance. |
| [`cornerstone-reference-07-home-upload-ask.png`](cornerstone-reference-07-home-upload-ask.png) | Home workspace | Drop zone, ask box, recent items, knowledge states, suggested next steps, and recent activity. |
| [`cornerstone-reference-08-action-dry-run-approval.png`](cornerstone-reference-08-action-dry-run-approval.png) | Action dry-run | Action preview, impacted objects, proposed changes, external calls, policy decision, risk, and approval state. |

## Implementation Boundary

Use these images to preserve the visual direction:

- light-first calm workspace;
- small standard navigation;
- prominent global search;
- original artifacts and evidence kept close to the task;
- clear trust, risk, approval, and audit states;
- admin power separated into admin context;
- action previews that show dry-run, impact, policy, risk, approval, and auditability before execution.

Do not use these images to bypass scenario-first implementation, CLI parity, deterministic verification, security review, human-required approval, or evidence bundle requirements.
