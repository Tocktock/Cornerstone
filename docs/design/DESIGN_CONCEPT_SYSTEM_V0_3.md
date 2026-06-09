# CornerStone Design Concept & System v0.3

Status: Draft for product/design confirmation  
Owner: JiYong / Tars  
Date: 2026-06-09  
Design doctrine: **Calm Surface. Deep Evidence. Safe Action.**

---

## 1. Design Decision

The approved direction is the light workspace mockup and the admin-only UI direction:

- **Primary product surface:** calm, document/workspace-like, light theme, low-friction, centered around drop, search, ask, recent work, and quiet evidence state.
- **Admin surface:** still calm, but more structured and operational: connectors, policies, roles, namespaces, audit logs, and setup status.
- **Rejected direction:** dark command-center / heavy enterprise cockpit / always-visible policy-risk-evidence dashboard.

CornerStone should feel like a trusted knowledge desk that can become an operations platform when needed.

---

## 2. Product Personality

### 2.1 One-line concept

**CornerStone is a calm operational knowledge workspace where anything can be dropped, found, trusted, and safely acted on.**

### 2.2 Emotional target

Users should feel:

1. **Safe** — nothing dangerous happens without preview and approval.
2. **Clear** — they always know what is saved, searchable, evidence-backed, or still draft.
3. **Fast** — first value starts from drop/search/ask, not modeling or configuration.
4. **Trusting** — evidence exists behind every important claim, but it does not overwhelm the main view.
5. **In control** — admin power exists, but is separated from normal knowledge work.

### 2.3 Avoided feelings

CornerStone must not feel like:

- a surveillance dashboard;
- a security operations war room;
- a generic chatbot;
- a complex BI tool;
- a connector admin panel as the first experience;
- a model/ontology editor before value exists.

---

## 3. Core Visual Concept

### 3.1 Surface metaphor

Use a **calm desk + evidence cabinet** metaphor:

- The main page is the desk: drop, ask, recent items, next steps.
- Evidence is nearby but not always open: chips, popovers, drawers.
- Admin is a cabinet: structured, governed, searchable, auditable.
- Action is a sealed envelope: preview, diff, approval, then execution.

### 3.2 Visual language

| Layer | Direction |
|---|---|
| Theme | Light-first, quiet, high whitespace |
| Structure | Left navigation + top search + main content + optional right rail |
| Shape | Soft rectangles, 8–12px radius, subtle borders |
| Color | Blue for navigation/primary, green for trusted/evidence-backed, amber for review/risk, red only for urgent/destructive |
| Typography | Clean SaaS/product typography, strong headings, readable metadata |
| Density | Medium density by default; admin can be denser |
| Motion | Minimal; only for state changes and progress |
| Data viz | Avoid charts unless they answer a real user question |

---

## 4. Product Information Architecture

### 4.1 Standard user navigation

Default left navigation should stay small:

1. Home
2. Search
3. Artifacts
4. Claims
5. Actions

Optional lower/secondary areas:

- Workflows
- Reports
- Settings

Do not show Ontology, Policy, Audit, Tool Registry, Key Management, and Connector internals to standard users by default.

### 4.2 Admin navigation

Admin mode can expose operational controls:

1. Workspaces
2. Users
3. Connectors
4. Policies
5. Settings
6. Audit logs

Admin UI should be visually related to the product UI, but more table/card driven and less conversational.

### 4.3 Global header

The global header should include:

- CornerStone logo
- Universal search: “Search across artifacts, claims, entities, and more…”
- Help
- Notifications
- User/account

The search bar is a product promise: everything important should become findable.

---

## 5. Page Concepts

### 5.1 Home — Universal Workspace

Purpose: first value and daily return point.

Default composition:

- Hero: “Drop anything, or ask what we know”
- Large drop zone
- Ask box
- Suggested prompts
- Recent items
- Knowledge states card
- Suggested next steps
- Right rail: recent activity

Primary rule: first screen must communicate **Drop → Search → Evidence** without explaining the full platform.

### 5.2 Search

Purpose: find artifacts, claims, entities, and actions in one place.

Default composition:

- Large query input
- Type filters: All, Artifacts, Claims, Entities, Actions
- Date/source/more filters
- Result list with type icon, title, snippet, owner, updated time, trust chips
- Right rail: “What we found,” top entities, suggested follow-ups

Primary rule: search results are not just links. They are reusable evidence candidates.

### 5.3 Artifact Viewer

Purpose: inspect original source and derived understanding.

Default composition:

- Breadcrumb and title
- Trust state chip
- Save / ask about this file / view linked evidence actions
- File metadata strip
- Original document preview
- Right panel: summary, extracted entities, related claims, provenance

Primary rule: original artifact is visually primary. Derived summaries are helpful, but not the source of truth.

### 5.4 Object Explorer

Purpose: understand a real-world object or entity without exposing raw ontology complexity.

Default composition:

- Entity profile sidebar
- Overview tabs: Overview, Related Artifacts, Claims, Connections, Activity
- “At a glance” cards
- Connections list with optional graph view
- Right rail: key facts, risk & trust, highlights

Primary rule: ontology appears as useful object context, not as schema engineering.

### 5.5 Claim Builder

Purpose: turn evidence into a human-readable conclusion.

Default composition:

- Claim title and trust ladder: Draft → Evidence-backed → Approved
- Claim statement field
- Rationale field
- Category/tags/framework links
- Right evidence picker
- Save Draft / Request Review / Promote to Decision

Primary rule: the user can draft freely, but cannot approve unsupported conclusions.

### 5.6 Evidence Details

Purpose: explain why a claim or action can be trusted.

Default composition:

- Claim header
- Evidence tabs: Supporting, Counter, Unresolved
- Evidence list with source, snippet, collection time, relevance, confidence
- Right drawer: source metadata, lineage, provenance, integrity, linked objects

Primary rule: evidence should be one click away, not always in the user’s face.

### 5.7 Action Studio — Dry Run

Purpose: show what will happen before anything changes.

Default composition:

- Action title
- State chips: Dry run, Approval required, Risk level
- Summary
- Impacted objects
- Proposed changes/diff
- External calls
- Policy decision
- Right rail: risk and approval details
- Save as draft / Request approval

Primary rule: action is never “magic automation.” It is a previewable, reviewable card.

### 5.8 Ops Inbox

Purpose: daily triage for items needing attention.

Default composition:

- Tabs: Needs review, Approval requests, Policy blocked, Failed runs
- Filterable list/table
- Right preview panel
- Next actions
- Internal note input

Primary rule: inbox is for attention and triage, not full investigation.

### 5.9 Workflow Run / Audit

Purpose: prove what happened.

Default composition:

- Workflow/run title and status
- Run metadata
- Timeline: Drafted → Dry run → Awaiting approval → Executed → Logged
- Run outputs
- Audit trail
- Right rail: policy decision, execution summary, rollback/compensation, linked items

Primary rule: the system must make completion auditable, not merely “successful.”

### 5.10 Admin — Connectors / Policies / Roles

Purpose: controlled setup and governance.

Default composition:

- Admin breadcrumb
- Tabs: Sources, Policies, Access roles, Namespace
- Connected sources list
- Namespace settings
- Right rail: policies, access roles, recent activity

Primary rule: admin power is visible only in admin context. Normal users should not feel admin burden.

---

## 6. Design System Foundations

### 6.1 Layout tokens

| Token | Value | Use |
|---|---:|---|
| App max width | Fluid | Full app shell |
| Sidebar width | 248px | Primary nav |
| Right rail width | 320–360px | Activity/evidence/admin context |
| Content gutter | 32px | Main page spacing |
| Card padding | 20–24px | Standard cards |
| Row height | 56–72px | Lists and tables |
| Radius small | 6px | Chips, inputs |
| Radius medium | 10px | Buttons, small cards |
| Radius large | 14px | Cards, panels |

### 6.2 Color roles

| Role | Intended color family | Meaning |
|---|---|---|
| Primary | Blue | Navigation, main action, active state |
| Evidence / trusted | Green | Evidence-backed, connected, validated |
| Review / caution | Amber | Under review, medium risk, approval required |
| Danger | Red | Destructive, failed, high risk |
| Neutral | Slate / blue-gray | Text, borders, panels |
| Background | Near-white | Calm workspace |

Important: color should supplement labels, never replace them.

### 6.3 Trust state chips

Required chips:

- Saved
- Searchable
- Draft
- Evidence-backed
- Corroborated
- Under review
- Insufficient evidence
- Approved
- Executed
- Failed
- Policy blocked

Trust chips must use plain language. Avoid internal names like “L3 Evidence” in normal UI.

### 6.4 Type scale

| Token | Use |
|---|---|
| Display | Home hero only |
| Page title | Main page titles |
| Section title | Cards and panels |
| Body | Regular text |
| Metadata | timestamps, source, owner, file type |
| Label | chips, form labels, table headers |

### 6.5 Icons

Icon language should be simple and semantic:

- File: artifact
- Shield/check: evidence-backed or trusted
- People: claim/collaboration
- Check circle: action or completed state
- Warning triangle: review/risk
- Link: lineage/reference
- Building: workspace/tenant/entity

Avoid decorative icon overload.

---

## 7. Component System

### 7.1 Core primitives

- AppShell
- SidebarNav
- TopSearch
- PageHeader
- Card
- Panel
- RightRail
- Drawer
- Table
- ListRow
- Tabs
- Button
- Input
- Textarea
- Select
- FilterButton
- DropdownMenu
- Badge/Chip
- Tooltip
- Toast
- Modal
- EmptyState
- FailureGuide

### 7.2 CornerStone semantic components

| Component | Purpose |
|---|---|
| DropZone | First-value artifact ingestion |
| AskBox | Question entry without making product chatbot-only |
| RecentActivityRail | Calm right-side operational context |
| KnowledgeStateCard | Saved/Searchable/Evidence-backed state summary |
| EvidenceChip | Quiet evidence/trust state marker |
| TrustLadder | Draft → Evidence-backed → Approved progress |
| EvidencePicker | Attach evidence to claims |
| EvidenceDrawer | Inspect source, metadata, provenance, lineage |
| SourceCard | Compact source/evidence unit |
| ClaimCard | Claim preview with trust state and evidence count |
| ActionPreviewCard | Dry-run action summary |
| PolicyDecisionPanel | Allowed/blocked/allowed with approval reasoning |
| ImpactDiff | Human-readable proposed changes |
| ApprovalPanel | Request/review/approve action state |
| AuditTimeline | Proven lifecycle and execution history |
| NamespaceSelector | Workspace/tenant context |
| ConnectorStatusRow | Admin connector status |
| AccessRoleCard | Admin permission overview |

---

## 8. Interaction Rules

### 8.1 Progressive disclosure

Default view shows the simplest safe state. Deeper evidence, policy, lineage, and audit appear when:

- the user clicks a chip;
- the claim is promoted;
- an action is previewed;
- an admin opens policy/audit context;
- a failure requires explanation.

### 8.2 Action safety

Action flow:

1. Suggested next step
2. Preview / dry run
3. Diff + impact + policy + risk
4. Approval if required
5. Execute
6. Audit log / result

Never show a primary “execute” action before preview and approval requirements are clear.

### 8.3 Evidence behavior

Evidence should appear in three levels:

1. **Chip:** “Evidence-backed,” “Supported by 4 artifacts”
2. **Popover/list:** supporting source names and snippets
3. **Drawer/detail:** provenance, lineage, integrity, linked objects

### 8.4 Admin separation

Admin-only controls must not leak into normal work pages except as read-only status chips. For example:

- Normal user sees: “Approval required”
- Admin sees: approval policy, rule source, scope, role mapping, audit event

---

## 9. Page-State Rules

Every major page must define these states:

- Empty
- Loading
- Ready
- Partial / degraded
- Needs review
- Permission denied
- Policy blocked
- Failed with recovery
- Audit/log available

Failure states must include:

1. What happened
2. Why it happened
3. What stayed safe
4. What the user can do next

---

## 10. Content Style

### 10.1 Voice

CornerStone should sound:

- calm;
- precise;
- evidence-aware;
- not overly excited;
- not apologetic unless the system failed;
- not pretending certainty.

### 10.2 Preferred copy patterns

Good:

- “Searchable”
- “Evidence-backed”
- “Supported by 4 artifacts”
- “Approval required before execution”
- “No external send yet”
- “Policy blocked: missing approval rule”

Avoid:

- “AI knows”
- “Guaranteed”
- “Autonomous execution ready”
- “Magic workspace”
- “Instant truth”

---

## 11. Admin UI Concept

Admin UI should be **structured confidence**, not complexity.

### 11.1 Admin visual rules

- Use the same light theme and component system.
- Use denser lists/tables than standard user pages.
- Use right rail cards for policies, roles, and recent changes.
- Always show scope: workspace, namespace, connector, policy bundle.
- Make dangerous configuration changes visibly reviewable.

### 11.2 Admin required surfaces

- Workspaces
- Users
- Connectors
- Policies
- Access roles
- Namespaces
- Settings
- Audit logs

### 11.3 Admin safety states

- Connected
- Needs setup
- Sync paused
- Policy enforced
- Approval required
- Read-only
- Egress denied
- Audit enabled
- Key rotation needed

---

## 12. Design Redlines

Do not ship UI that violates these:

1. Do not make the first screen an admin dashboard.
2. Do not make the first screen a chatbot only.
3. Do not show ontology/schema setup before first value.
4. Do not show full policy/risk/audit panels by default on Home.
5. Do not use dark command-center style as the default theme.
6. Do not rely only on color for trust/risk state.
7. Do not allow claims to look approved without evidence.
8. Do not show execute as the primary action before dry-run and approval status.
9. Do not hide workspace/tenant context.
10. Do not describe generated output as truth without evidence state.

---

## 13. Design Acceptance Scenarios

| ID | Scenario | Expected UI behavior |
|---|---|---|
| DS-S01 | New user opens CornerStone | Sees drop zone, ask box, recent items, and simple knowledge state summary |
| DS-S02 | User uploads a file | File is saved first, then becomes searchable or partial with helpful status |
| DS-S03 | User searches | Results include artifacts/claims/entities/actions with trust chips and filters |
| DS-S04 | User opens artifact | Original preview is primary; derived summary and provenance are available |
| DS-S05 | User drafts claim | Claim starts Draft; evidence picker is visible; approval path is clear |
| DS-S06 | Claim has no evidence | It cannot appear as approved; UI shows Draft or Insufficient evidence |
| DS-S07 | User previews action | Dry-run card shows diff, impact, policy decision, risk, approval requirement |
| DS-S08 | Action executes | Workflow run shows timeline, outputs, audit trail, linked items |
| DS-S09 | Admin opens connectors | Shows sources, policies, roles, namespace, recent admin activity |
| DS-S10 | Policy blocks action | UI explains cause, safety result, and resolution path |

---

## 14. Implementation First Slice

Start with a small UI kit, not the full platform.

### 14.1 Must-build first

1. AppShell
2. SidebarNav
3. TopSearch
4. Home / Universal Workspace
5. DropZone
6. AskBox
7. RecentItemsList
8. KnowledgeStateCard
9. RecentActivityRail
10. TrustStateChip
11. EvidenceChip
12. ActionSuggestionRow
13. AdminShell
14. ConnectorStatusRow
15. PolicyCard

### 14.2 Build later

- Graph view
- Advanced ontology editor
- Full workflow builder
- Custom dashboard builder
- Full report designer
- Dark theme
- Dense analyst mode

---

## 15. Final Concept Lock Candidate

**Name:** CornerStone Calm Operations UI  
**Doctrine:** Calm Surface. Deep Evidence. Safe Action.  
**Default page:** Universal Workspace  
**Default theme:** Light  
**Default navigation:** Home, Search, Artifacts, Claims, Actions  
**Admin mode:** Workspaces, Users, Connectors, Policies, Settings, Audit logs  
**Core states:** Saved, Searchable, Evidence-backed, Draft, Under review, Approved, Dry run, Approval required, Executed, Policy blocked  
**Core promise:** Anything can be dropped, found, trusted, and safely acted on.
