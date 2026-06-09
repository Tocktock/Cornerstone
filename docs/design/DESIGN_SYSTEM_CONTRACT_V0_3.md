# CornerStone Design System Contract v0.3

**Status:** Active design-system contract for future UI implementation.
**Owner:** JiYong / Tars.
**Source documents:**
- `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md`
- `docs/design/tokens/cornerstone_design_tokens_v0_3.json`
**Doctrine:** Calm Surface. Deep Evidence. Safe Action.

## 1. Purpose

This contract turns the supplied design concept, design tokens, and reference images into repo-level implementation guidance.

It does not implement UI code. It defines what future CornerStone UI work must preserve before it can claim product-design readiness.

## 2. Reference Image Reading

The reference images show a light, calm product shell with two related but distinct operating modes:

1. **Universal workspace and scenario surfaces**
   - Left sidebar navigation stays small and understandable: Home, Search, Artifacts, Claims, Actions, Workflows, Audit.
   - The top search bar is global and prominent.
   - Home starts from drop, ask, recent items, and knowledge-state summary.
   - Search results mix artifacts, claims, entities, and actions with trust chips and evidence labels.
   - Artifact viewing keeps the original document visually primary while evidence, provenance, related items, and permissions remain nearby.
   - Claim, action, dry-run, approval, workflow, and audit surfaces use simple panels, not command-center density.

2. **Admin connector/policy surface**
   - Admin has the same calm shell, but exposes Workspaces, Users, Connectors, Policies, Settings, and Audit logs.
   - Connector management is table/list driven with right-rail policy, access role, and activity panels.
   - Scope and governance are visible: namespace, source, policy, access role, recent activity.
   - Admin power is visually contained inside admin context and does not become the default user experience.

The images confirm the product direction: first value is a calm workspace; evidence, policy, approval, and audit are close at hand but progressively disclosed.

## 3. Design Authority Order

For UI, design-system, or component work, read in this order:

1. Product SoT and scenario standard.
2. `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`.
3. `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md`.
4. `docs/design/tokens/cornerstone_design_tokens_v0_3.json`.
5. Task-specific scenario contract and implementation code.

If a lower-priority visual idea conflicts with evidence-first product rules, safety rules, or CLI/native verification rules, the product and safety rules win.

## 4. Non-Negotiable Visual Direction

CornerStone must feel like:

- a trusted knowledge desk;
- a calm operational workspace;
- a place where evidence is available without overwhelming the main task;
- a system that previews, explains, and audits actions before execution.

CornerStone must not feel like:

- a dark command center;
- a surveillance dashboard;
- a generic chatbot;
- a dense BI cockpit;
- a connector admin product as the first experience;
- a model, ontology, or policy editor before first value.

## 5. Token Requirements

The canonical token source is `docs/design/tokens/cornerstone_design_tokens_v0_3.json`.

Required token groups:

- `color`
- `space`
- `radius`
- `layout`
- `shadow`
- `typography`
- `state`

Required state tokens:

- `saved`
- `searchable`
- `draft`
- `evidenceBacked`
- `corroborated`
- `underReview`
- `insufficientEvidence`
- `approved`
- `executed`
- `failed`
- `policyBlocked`

Implementation must preserve semantic color roles:

| Role | Meaning |
|---|---|
| Blue | Navigation, primary action, active state |
| Green | Evidence-backed, connected, validated, executed |
| Amber | Review, caution, approval required, insufficient evidence |
| Red | Failed, blocked, destructive, urgent |
| Slate / blue-gray | Text, borders, panels, metadata |

Color must supplement text labels. Trust, risk, and policy states cannot rely on color alone.

## 6. Required Product Surfaces

Future UI work must preserve these design contracts:

| Surface | Contract |
|---|---|
| Home / Universal Workspace | Drop, ask, recent work, and knowledge states are the first-value path. |
| Search Results | Search spans artifacts, claims, entities, and actions with filters and trust chips. |
| Artifact Viewer | Original artifact is primary; derived summary, provenance, permissions, and related items are secondary. |
| Claim Builder | Drafting is free; approval requires evidence visibility and trust-state clarity. |
| Action Studio | Mutations start as preview/dry-run with diff, impact, policy, risk, and approval state. |
| Approval Center | Approval requests explain why approval is required and who must approve. |
| Workflow Runs | Run status is timeline-based and auditable. |
| Audit & Evidence | Events, policy decisions, evidence bundles, and export actions are inspectable. |
| Admin Console | Connectors, policies, roles, namespaces, and audit logs are structured and scoped. |

## 7. Component Baseline

The first UI kit should start with:

- `AppShell`
- `SidebarNav`
- `TopSearch`
- `PageHeader`
- `DropZone`
- `AskBox`
- `RecentItemsList`
- `KnowledgeStateCard`
- `RecentActivityRail`
- `TrustStateChip`
- `EvidenceChip`
- `ActionSuggestionRow`
- `AdminShell`
- `ConnectorStatusRow`
- `PolicyCard`

Do not begin with graph views, ontology editors, dashboard builders, or dark theme.

## 8. Interaction Rules

Use progressive disclosure:

1. Main page shows the simplest safe state.
2. Evidence appears as chip, popover/list, then drawer/detail.
3. Policy and audit detail appear when action, approval, failure, or admin context requires them.
4. Mutating actions must show dry-run, impact, policy decision, and approval status before execution.

Every major page must define:

- empty state;
- loading state;
- ready state;
- partial/degraded state;
- needs-review state;
- permission denied;
- policy blocked;
- failed with recovery;
- audit/log available.

Failure states must say what happened, why it happened, what stayed safe, and what the user can do next.

## 9. Content Rules

Use calm, precise, evidence-aware copy.

Preferred:

- `Searchable`
- `Evidence-backed`
- `Supported by 4 artifacts`
- `Approval required before execution`
- `No external send yet`
- `Policy blocked: missing approval rule`

Avoid:

- `AI knows`
- `Guaranteed`
- `Autonomous execution ready`
- `Magic workspace`
- `Instant truth`

## 10. Design Acceptance Scenarios

The source design concept defines `DS-S01` through `DS-S10`. They are design acceptance scenarios and should be referenced in future UI work alongside product scenarios.

Minimum initial design gates:

| ID | Gate |
|---|---|
| DS-S01 | New user sees drop zone, ask box, recent items, and knowledge state summary. |
| DS-S03 | Search results include artifacts, claims, entities, actions, trust chips, and filters. |
| DS-S04 | Artifact viewer keeps original preview primary. |
| DS-S05 | Claim builder shows evidence picker and trust ladder. |
| DS-S07 | Action preview shows dry-run, diff, impact, policy, risk, and approval state. |
| DS-S09 | Admin connector page separates sources, policies, roles, namespace, and recent activity. |

## 11. Verification Rule

Design-system documentation edits must pass:

```sh
scripts/verify_design_system_docs.sh
scripts/verify_sot_docs.sh
```

Future UI implementation cannot claim design PASS without:

- screenshots or browser traces for the relevant design scenarios;
- token usage or token mapping evidence;
- state coverage for empty/loading/ready/degraded/blocked/failed;
- accessibility and responsive checks;
- explicit `HUMAN_REQUIRED` marking for subjective visual approval.
