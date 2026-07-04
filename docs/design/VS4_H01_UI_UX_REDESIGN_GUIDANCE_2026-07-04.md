# VS4-H01 UI/UX Redesign Guidance

**Date:** 2026-07-04 KST
**Owner:** JiYong / Tars
**Status:** Proposed — awaiting owner adoption. Becomes the working design brief for the VS4-H01 UI Recovery effort once accepted.
**Role:** This document is the full-product redesign direction requested after the `VS4-H01 = REJECT` decision. It fulfills and extends the Phase 1 artifact recommended by `docs/verification-reports/VS4_H01_UI_UX_REJECTION_REPORT_2026-07-04.md` (reference mapping), and adds product-wide principles, per-surface guidance, and an implementation plan.
**Authority position:** Subordinate to `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`, `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md`, and `docs/design/tokens/cornerstone_design_tokens_v0_3.json`. This document does not change the design doctrine; it operationalizes it against the current implementation. Where it proposes something the v0.3 set does not cover (notably the Brief surface), the addition is labeled **Proposed** and requires owner approval.

Evidence labels used throughout: **Verified** (observed in code, screenshots, or live records), **Specified** (required by the v0.3 design set or SoT), **Proposed** (this document's recommendation).

---

## 1. Executive Summary

The VS4 Product Alpha UI was rejected for good reason, and the rejection is not fixable with styling changes. The current "product" is a single ~6,300-line Python function emitting one HTML page in which all seven product surfaces are stacked as scroll sections, navigation is anchor links, the design tokens are unused, and the page's first messages are internal verification flags. The UI is best understood as a verification fixture that humans are allowed to look at — not as a product surface that verification happens to check.

The redesign therefore has one thesis: **rebuild the presentation layer as a small multi-page product UI on top of the existing runtime, API, and audit substrate — wired to the existing v0.3 design tokens, speaking only product language, populated only with real local data.** The substrate (artifact store, trust states, audit ledger, JSON API, browser-proof harness) is sound and stays. The page-generation approach, the information architecture, the visual layer, and the copy are replaced.

This is deliberately not a "make Home prettier" plan. Section 4 gives direction for every user-facing surface; Section 9 sequences the work so the owner can re-review early.

---

## 2. Honest Assessment — Why the Current UI/UX Fails

The rejection report's gap analysis (visual hierarchy, IA, first-value workflow, progressive disclosure, reference usage, verification gap) is accurate. This section goes one level deeper: the six root causes below explain *why* those gaps appeared and why polish cannot close them.

### RC1 — The UI is a test fixture first, a product second (Verified)

The Home page is structured around scenario observability, not user tasks. Sections exist because scenario rows assert `data-vs4-*` markers on them (`packages/cornerstone_cli/product_runtime.py:1565-1700`); panel copy exists to satisfy absence/presence checks ("Package is not acceptance.", "No acceptance collected"). The header renders `local_scenario_ready=true`, `vs0_runtime_ready=true`, `production_release_ready=false`, `real_external_http_calls=0` as user-visible badges. On mobile, those flags are the entire first viewport before the user sees a single action (Verified: `reports/browser/vs4-product-alpha-ui-daily-loop-slice-001-mobile/home.png`).

Consequence: the page optimizes for the verifier's reading order, not the user's. No amount of restyling changes what the page is *about*.

### RC2 — There are no pages (Verified)

`PRODUCT_ALPHA_NAV` maps Home/Search/Artifacts/Claims/Actions to `#anchors` inside one document (`product_runtime.py:71-77`). All seven `UI_SURFACES` are sections of a single scroll. The JSON API has per-record routes, but no HTML page exists for any record, list, or task other than the monolith.

Consequence: information architecture and progressive disclosure are *structurally impossible*. Progressive disclosure requires somewhere to disclose *into*; a one-page product can only pile everything onto the first screen — which is exactly what happened. This is the single most important fix.

### RC3 — The design system exists on paper only (Verified)

`cornerstone_design_tokens_v0_3.json` defines the palette (primary blue `#2563EB`, app background `#F8FAFC`, Inter type ramp, 11 semantic state chips). The implementation uses a hand-rolled `:root` with a different palette (teal accent `#285e61`, `#f7f9fb` paper), a different font stack, and a 3-state badge system (`safe`/`warn`/default) (`product_runtime.py:688-698`). Layout tokens (248px sidebar, 340px right rail, 72px header) are not used; the reference shell was never built.

Consequence: even where the right words appear ("Drop", "Ask", "Evidence-backed"), the screen cannot look like the reference, because none of the reference's visual constants exist in the code. Tokens were archived, not adopted.

### RC4 — The product speaks repo dialect (Verified)

User-facing surfaces render internal vocabulary directly: snake_case policy states as chips (`review_before_authority`, `evidence_required_before_approval`, `draft_memory_review_required`), raw record IDs (`evidence_bundle:evb_b7551adb2b68e772`, `audit:audit_fa514ee403b85465`), scenario IDs (`VS4-H01`), and review-process nouns ("review packet", "package paths and commands", "Ready for JiYong/Tars walkthrough") — all visible in `reports/browser/vs4-product-alpha-ui-daily-loop-slice-001/home.png`. The design contract's content rules (§9) specify plain-language chips like "Approval required before execution"; the concept (§6.3) explicitly bans internal names in normal UI (Specified).

Consequence: a normal user must learn the repo's internal ontology to parse the first screen. This is the "the page asks the user to understand the system before using it" failure in its most literal form.

### RC5 — Safety honesty metastasized into the main content (Verified)

Scope honesty ("local mode, no external writeback, nothing is production-ready") is a real obligation. The current UI states it as: four header badges, a "Verification details" disclosure, a dedicated "Local Product Alpha" boundary panel, a "Learning boundary" panel, a "Human review handoff" panel, a "Review packet" panel, and negative-claim `data-*` attributes rendered as visible text throughout. The rejection report's acceptance checklist requires scope honesty "without dominating the page" (Specified).

Consequence: the product spends its first screen denying claims nobody made. Honesty is a footer sentence and a chip — not a wall.

### RC6 — Reference images were archived, not translated (Verified, matches report)

The reference set defines complete layouts for eight surfaces. Because references are correctly barred from being PASS evidence, no implementation requirement was ever derived from them — so the build satisfied scenario markers while ignoring the layouts. The report calls this out; the deeper lesson is that **design fidelity was never a first-class requirement anywhere in the verification system**, so the system converged on what it *did* measure.

### RC7 — The reference set itself has a hole where the product's core value lives (new finding, Verified absence)

Per `docs/adr/ADR-0007-product-value-first-reset.md`, the active spine is `Drop / Ask → Evidence-backed Brief → Decision → Audit`, and VS5's entire proof point is a stranger trusting a citation-grounded Brief. There is **no reference image for the Brief surface** — the eight references cover home, inbox, admin, search, claim, artifact, vendor, and action, but not the one surface VS5 external users will judge hardest. The current Brief "preview" is an empty stub panel (Verified: `product_runtime.py:1626-1630`).

Consequence: recovering only the referenced surfaces would still leave the spine's centerpiece undesigned. Section 4.2 proposes that direction; it needs explicit owner approval since no reference exists.

### Symptom → root cause → response (summary)

| Symptom (from rejection) | Root cause | Design response | Acceptance |
| --- | --- | --- | --- |
| Many panels compete on first screen | RC1, RC2 | Home eviction map (§4.1); real pages (§3) | REDG-01, REDG-02 |
| Internal labels/states shown early | RC4, RC5 | Language system + forbidden lexicon (§5.3) | REDG-03 |
| Looks like verifier dashboard, not workspace | RC1, RC3 | Token-wired shell + reference layouts (§4, §5) | REDG-01, REDG-08 |
| Review/package content on user Home | RC1, RC5 | Owner Review page (§4.9) | REDG-02 |
| Drop/Ask not dominant | RC2, RC6 | Home first-viewport contract (§4.1) | REDG-01 |
| Mobile shows flag wall first | RC1, RC5 | Mobile hierarchy rules (§7) | REDG-07 |
| Structural PASS masked design failure | RC6 | Design-fidelity gates in CI (§10) | REDG-09 |

---

## 3. Target Information Architecture

### 3.1 From one page to a small set of real pages (Proposed)

The UI becomes server-rendered HTML pages on the existing stdlib server — no SPA, no new framework, no new dependencies (see §9 constraints). The existing JSON API routes stay untouched; HTML page routes are added alongside them (exact prefix is an implementation decision; recommendation: HTML at `/` and `/{surface}` paths, JSON API stays where it is — decision D4).

| Route (Proposed) | Surface | Replaces (current monolith section) |
| --- | --- | --- |
| `/` | Home — Universal Workspace | `#home-ops-inbox` hero + drop/ask |
| `/search?q=` | Search results | `#search` section |
| `/artifacts`, `/artifacts/{id}` (HTML) | Artifact list + viewer | `#artifact-viewer` section |
| `/briefs/{id}` (HTML) | Brief detail (spine centerpiece) | `#vs4-brief-preview` stub |
| `/claims`, `/claims/{id}` (HTML) | Claim list + builder/detail | `#claim-builder` section |
| `/actions`, `/actions/{id}` (HTML) | Action list + dry-run card | `#action-card` section |
| `/inbox` | Ops Inbox (lanes + list + preview) | Ops Inbox panel on Home |
| `/audit` | Audit & evidence events (humanized) | `#audit-detail` section |
| `/review` | Owner Review Mode (all verification/handoff content) | header flags, handoff panel, review packet, readiness details |

### 3.2 Navigation model (Specified + Proposed)

- **Primary sidebar (exactly five):** Home, Search, Artifacts, Claims, Actions — per concept §4.1 (Specified).
- **Secondary (bottom of sidebar, visually quiet):** Inbox, Audit, Owner review (Proposed). Concept §4.1 allows optional secondary areas; Audit and internals must not sit in primary nav (Specified). Inbox stays secondary until VS6 makes it central (decision D2).
- **Global header:** logo, universal search ("Search across artifacts, claims, entities, and more…"), help, workspace label. No readiness flags, no scenario badges (Specified: concept §4.3).
- **Workspace context:** one quiet label ("Personal / default"). Do not build tenancy UI; do not imitate the reference's fictional "Acme Operations" org switcher (see §5.4 honesty rules).

### 3.3 Home eviction map (Proposed)

Everything currently on Home, with its destination. This is the concrete version of the report's Phase 3.

| Current Home content | Destination | Rationale |
| --- | --- | --- |
| Readiness badges (`local_scenario_ready=…` etc.) | `/review` | Internal state; RC4/RC5 |
| "Verification details" disclosure in header | `/review` | Internal state |
| Human review handoff panel + checklist | `/review` | Review process, not user task |
| Review packet + package paths + command list | `/review` | Review process |
| Learn review / "Learning candidate" panel | `/inbox` lane ("Needs review") | It is triage work, so it lives in the inbox |
| Ops Inbox lanes + rows + continue links | Compact "Continue work" module on Home (top 3, real items) linking to `/inbox` | Home shows the *entry* to triage, not the whole triage surface |
| "General-purpose packs / Same loop, different work" panel | Cut from UI (keep CLI demo) | Scenario demo content, not user value |
| "Local Product Alpha" boundary panel + "Learning boundary" panel | One footer sentence + one "Local" chip (§5.4) | Honesty without domination |
| Brief preview stub | Recent items list (real Briefs, once VS5 produces them) | Stub adds noise today |
| Drop textarea + Ask input | Rebuilt as reference-grade DropZone + AskBox (§4.1) | First value |

---

## 4. Page-by-Page Guidance

Format per surface: purpose → first-viewport contract (must show) → must-not-show → layout → interactions/states → mobile note. Layout constants come from tokens (§5.1): sidebar 248px, right rail 340px, header 72px, content gutter 32px.

### 4.1 Home — Universal Workspace (reference: `cornerstone-reference-07-home-upload-ask.png`)

**Purpose:** first value and daily return point. A new user must understand "I can drop anything, or ask what the system knows" within five seconds.

**First-viewport contract (desktop, in priority order — all Specified by reference + concept §5.1):**
1. App shell: sidebar (5 items), global search, quiet header.
2. Headline (Display type, the only Display on any page): "Drop anything, or ask what we know" + one supporting sentence.
3. Large drop zone: drag-and-drop target + "Browse files" + paste affordance. It must accept a real drop (client-side FileReader → existing `POST /artifacts` ingest; text-first is fine and honest).
4. Ask box directly below ("or ask a question"), single input + submit, wired to the real ask flow.
5. Up to three suggested prompts, seeded from the *actual* local corpus (else hidden).

**Below the fold / right rail:** Recent items (real artifacts/briefs, newest first) · Knowledge states card (real counts of Saved / Searchable / Evidence-backed) · Suggested next steps (derived from real inbox items; hidden when none) · Right rail: Recent activity (humanized audit events, e.g. "Brief created from vendor-renewal note — 2h ago").

**Must not show (anywhere on Home):** scenario/verifier vocabulary, readiness flags, package paths, raw record IDs, review-process panels, approval-policy internals — the full forbidden lexicon in §5.3. (Specified: rejection report Phase 2 + contract redline 4.)

**Day-zero state (this is the real first-run experience):** zero counts, empty recents. The empty Home is: headline, drop zone, ask box, and a knowledge-states card that says "Nothing here yet — drop your first file or note." Nothing else. Do not pad with placeholder content (§5.4).

**Boundary honesty:** one footer line — "Local mode: everything stays on this machine. Actions remain previews until you approve them." — plus a single "Local" chip in the header. That is the entire safety messaging on Home.

**Mobile:** headline → drop → ask → recents → states, in that order; nav collapses; right rail stacks last; zero horizontal overflow.

### 4.2 Brief Detail — the spine's centerpiece (no reference image; **Proposed**, needs owner approval — decision D1)

**Purpose:** present an evidence-backed Brief a stranger can read, trust, and act on in ten minutes. This is the surface VS5's external test lives or dies on, and the surface CS-VAL scenarios (grounding, citation integrity, uncertainty honesty) will be judged against.

**Composition (top to bottom):**
1. Title = the user's question or dropped source name; metadata line (created, source count); one trust chip reflecting *earned* state (per ADR-0007 §7, templated/extractive output must not carry "Evidence-backed" — if the generation path was fallback, the chip says "Draft" or "Keyword summary", honestly).
2. **Summary** ("What we found"): 2–4 sentences.
3. **Findings list:** each finding is a card with the statement, inline citation chips `[1][2]`, and a quiet confidence/trust chip. Citation chip → popover (source snippet + artifact link) → drawer (full provenance: fingerprint, ingest time, lineage). Three-level disclosure per contract §8.
4. **Gaps & uncertainty** ("What this brief cannot confirm"): honest, first-class section — not an apology, a feature.
5. **Suggested next steps:** buttons that create *drafts* (Draft claim from finding, Save to memory as candidate, Draft action) — never direct commits.
6. **Provenance footer (quiet):** model name + generated-at + link to evidence bundle and audit trail.

**Right rail:** Sources used (each with trust chip and link) · Related items.

**Must not show:** raw bundle/audit IDs in the body (they live in the drawer/audit page), model plumbing, prompt text, scenario language.

**Failure/degraded states:** generation failed → show the four-line failure pattern (§6.3) with the extractive fallback clearly labeled "Keyword summary (no model)"; never silently downgrade with an unearned chip.

### 4.3 Search (reference: `cornerstone-reference-04-search-results.png`)

Query input carried into the page · type tabs with real counts (All / Artifacts / Claims / Entities / Actions) · result rows: type icon, title, honest snippet (current engine is keyword-based — snippets are matched text, no semantic claims), owner/date metadata, trust chips · right rail "What we found" counts only when non-zero · empty state suggests broadening or dropping content. Filters limited to what the backend really supports today (date, type); do not render dead filter chrome (Proposed; reference shows more filters than the engine has — see §5.4).

### 4.4 Artifact Viewer (reference: `cornerstone-reference-06-artifact-viewer.png`)

Breadcrumb + title + trust chip + actions ("Ask about this file", "View linked evidence") · metadata strip (source, ingested, type, workspace) · **original content primary** (text-first viewer is honest for current ingest; no fake PDF chrome) · right panel: Summary (only when a real one exists), Extracted keywords (labeled as keywords — the current extractor is a tokenizer, not entity intelligence; do not label it "Extracted entities" until it is one), Related claims, Provenance (truncated SHA256 with copy, full detail in drawer). Specified rule: original artifact visually primary; derived content secondary (contract §6, DS-S04).

### 4.5 Claims — list + builder (reference: `cornerstone-reference-05-claim-draft-supporting-evidence.png`)

Builder: claim statement + rationale fields · **trust ladder visible** (Draft → Evidence-backed → Approved) · evidence picker in right rail (real evidence bundles/artifacts only) · Save draft / Request review actions · approval affordance disabled until evidence is attached, with plain-language explanation ("Add evidence before approval") rather than the raw `evidence_required_before_approval` state (Specified: DS-S05, DS-S06; redline 7).

### 4.6 Actions — list + dry-run card (reference: `cornerstone-reference-08-action-dry-run-approval.png`)

State chips: "Preview (dry run)" / "Approval required" / risk level · summary → impacted objects → proposed change (diff-style) → external calls ("Planner task — will be created *(simulated in local mode)*") → policy decision ("Allowed with approval") · right rail: risk & approval · primary button is "Request approval", never "Execute", until approval state is satisfied (Specified: concept §8.2, redline 8). The local/mock boundary appears once, inline at the external-calls row — not as a page banner.

### 4.7 Ops Inbox (reference: `cornerstone-reference-02-operations-inbox.png`)

Lane tabs with real counts (Needs review / Approval requests / Policy blocked / Failed runs) · filterable list rows: type icon, title, one-line description, plain-language status chip, time · desktop: right preview panel with next actions; row click opens the record's own page · learning candidates appear here as "Needs review" items (from Home eviction). Inbox is triage, not investigation (Specified: concept §5.8).

### 4.8 Audit (secondary surface)

Humanized event list ("Artifact ingested", "Brief created", "Approval recorded" with relative times), each row expandable to the raw event (IDs, hashes, chain position). Raw detail is one level down, never the list's first reading. Reachable from provenance footers and the sidebar's secondary area.

### 4.9 Owner Review Mode — `/review` (Proposed)

Everything evicted from Home lands here, intact: readiness flags, scenario/human-gate status, walkthrough checklist, review packet, package paths and commands, negative-claim details, links to verification reports. Framed with a plain header: "Owner review — internal readiness and acceptance records. Not part of the product surface." Existing `data-vs4-*` markers for these regions move with their content so Plane-1 scenarios can re-anchor here (coordinated change — decision D3). This page is allowed to be dense; it is for the owner and verifier, and its existence is what lets every other page be calm.

### 4.10 Deliberately out of scope (Specified: ADR-0007 dormancy register)

Vendor/entity Object Explorer (`reference-01`), Admin Connectors (`reference-03`), workflow builder, graph views, ontology surfaces, dark theme. These references remain archived direction; building them now would violate the scope freeze. The redesign must not add navigation entries for them.

---

## 5. Visual System and Language System

### 5.1 Tokens become law (Specified)

- Generate CSS custom properties **from** `cornerstone_design_tokens_v0_3.json` at server start (one small helper; tokens JSON stays the single source). Delete the hand-rolled `:root` palette.
- Apply the layout tokens: 248px sidebar, 340px right rail, 72px header, 32px gutter, 24px card padding, 64px list rows.
- Type ramp from tokens: Display 40/48 (Home hero only), Page title 32/40, Section title 16/24, Body 14/22, Metadata 12/18, Label 12/16. Font stack per tokens (`Inter, ui-sans-serif, system-ui, …` — system fallback is fine; do not fetch remote fonts).
- Color roles per contract §5: blue = navigation/primary action, green = evidence/validated, amber = review/caution, red = failed/blocked/destructive only, slate = text/borders. Enforcement rule: **no hex literal outside the token pipeline** (lintable).

### 5.2 One chip system (Specified)

Eleven semantic chips, styled exactly from `state` tokens, labels in plain language: Saved · Searchable · Draft · Evidence-backed · Corroborated · Under review · Insufficient evidence · Approved · Executed · Failed · Policy blocked. Color never carries meaning alone; every chip has its text label (contract §5).

### 5.3 Language map and forbidden lexicon (Proposed, enforcing contract §9)

Internal state → user-facing phrase (rendering layer owns this mapping; internal values stay in APIs and `/review`):

| Internal | User surface phrase |
| --- | --- |
| `review_before_authority` | "Review required before approval" |
| `evidence_required_before_approval` | "Add evidence before approval" |
| `draft_memory_review_required` | "Review before saving to memory" |
| `local/mock`, `real_external_http_calls=0` | "Simulated in local mode" (inline, where an external call is shown) |
| `evb_…`, `audit_…`, `brief_…` raw IDs | Human titles; IDs only in provenance drawer / audit detail |

**Forbidden on product surfaces** (automatable as an absence check; `/review` exempt): snake_case state strings; `VS[0-9]`/scenario/gate IDs; "scenario", "verifier", "human gate", "acceptance", "walkthrough", "package path", "readiness", "browser proof", "review packet"; raw record-ID prefixes; boolean flag text (`…=true/false`). This list seeds REDG-03's scan and should live next to the existing sensitive-marker scan.

### 5.4 Honesty rules — reference fiction vs. product truth (Proposed; important)

The reference images contain fictional content the product cannot truthfully render: teammates and avatars ("Sarah Chen", "+2"), org tenancy ("Acme Operations"), thousands-scale counts ("3,842 Searchable"), vendor registries. **Adopt the references' structure, hierarchy, and visual language — never their fictional data.** Concretely:

1. Render only records that exist in the local store. No seeded fake users, fake counts, fake activity.
2. Design the day-zero empty state as the primary first-run experience (it is what VS5 strangers may first see).
3. Single-user truth: no avatars, no presence dots, no collaborator affordances until multi-user exists.
4. No dead chrome: controls render only when wired to a real behavior (filters, "View all", suggested prompts).

Rationale: fabricated fullness would contaminate VS5 external evidence more than a sparse honest screen — testers would react to theater. This rule keeps reference alignment compatible with the repo's evidence discipline.

---

## 6. Interaction Rules (Specified, consolidated from contract §8 / concept §8)

1. **Disclosure ladder:** chip → popover/list → drawer → dedicated page. Nothing deeper than one level appears by default. Evidence: "Evidence-backed" chip → sources popover → provenance drawer → audit page.
2. **Action safety sequence:** suggestion → preview/dry-run (diff + impact + policy + risk) → approval if required → execute → audit record. Execute is never the primary button before preview and approval state are visible.
3. **Failure pattern (every failure, everywhere):** what happened · why · what stayed safe · what to do next. Falling back (e.g., model unavailable → keyword summary) re-labels the output's trust chip; no unearned labels.
4. **Page states:** every page implements empty / loading / ready / degraded / needs-review / permission-or-policy-blocked / failed-with-recovery as designed states, with day-zero emphasized (§5.4).
5. **Keyboard & a11y:** keep the existing skip link and strong focus outlines (Verified good); logical tab order Home → search → drop → ask → recents → next steps (matches retry criterion 5's focus path); all disclosure elements keyboard-operable; touch targets ≥ 44px.

---

## 7. Responsive Rules

- Breakpoints: single column below 760px (existing breakpoint is fine); right rail stacks below main content from 760–1200px; full three-zone shell ≥ 1200px (Proposed).
- **Mobile first viewport = same hierarchy as desktop, minus the rail: headline, drop, ask.** Never status chips first (kills the current badge-wall, RC5). Mobile removes and reorders; it never adds content desktop doesn't have.
- No body-level horizontal overflow on any page (retry criterion 7). Wide tables/diffs scroll inside their own container.
- Automated proof: desktop + mobile screenshots per surface, plus a `scrollWidth <= clientWidth` body assertion in the browser proof (extends the existing Chrome-CDP harness).

---

## 8. Consistency Rules

1. **One shell.** Every page renders through one shared shell (sidebar + header + content + optional rail). No page builds its own frame.
2. **One component per problem.** One card, one chip set, one list row, one drawer, one empty-state pattern, one failure pattern — reused everywhere. Component baseline = contract §7 list (AppShell, SidebarNav, TopSearch, PageHeader, DropZone, AskBox, RecentItemsList, KnowledgeStateCard, RecentActivityRail, TrustStateChip, EvidenceChip, ActionSuggestionRow); implement as Python render helpers in one module, not per-page copies.
3. **Spacing/radius/color only from tokens.** No ad-hoc values; lint for hex literals and px values outside the scale.
4. **Copy from the approved pattern list** (contract §9). New recurring phrases get added to the language map (§5.3), not invented per page.
5. **Markers are metadata.** `data-*` verification markers are allowed everywhere but must never be rendered as visible text or drive layout. Verifiers read attributes; humans read the page. (This decouples Plane-1 proof from design — the fix for RC1.)

---

## 9. Implementation Plan

**Standing constraints (all increments):** stdlib-only server stays; no new runtime dependencies or frontend toolchain (scope freeze until VS5 closes — ADR-0007 §6); existing JSON API and CLI behavior untouched; audit/ingest/trust substrate untouched; every increment keeps `verify_design_system_docs.sh` / `verify_sot_docs.sh` green and regenerates affected browser proofs.

| Inc. | Goal | Key work | Exit evidence | Strongest claim |
| --- | --- | --- | --- | --- |
| **R0** | Foundation: tokens + shell + routing | Token→CSS pipeline; shared shell/component render module (new `product_ui` module, `product_runtime.py` keeps serving); HTML routes scaffolded; language-map helper | Token parity check (rendered CSS ⊇ tokens); shell screenshot; routes return 200 | STRUCTURAL_READY |
| **R1** | Home rebuilt to reference | §4.1 composition incl. real drag-drop; eviction map applied; day-zero state; `/review` created to receive evicted content with markers | Desktop+mobile+day-zero screenshots; REDG-01/02/03 pass; informal owner directional look | Home design-implemented (local) |
| **R2** | Spine pages | Brief detail (§4.2), Search (§4.3), Artifact viewer (§4.4) as real pages wired to real records | Per-page screenshot pairs; DS-S03/S04 checks; citation disclosure ladder demo on a real brief | Spine surfaces implemented (local) |
| **R3** | Work pages | Claims builder (§4.5), Action dry-run (§4.6), Inbox (§4.7), Audit (§4.8) | DS-S05/S06/S07 checks; screenshots | All surfaces implemented (local) |
| **R4** | Language & chip sweep | Chip system everywhere; forbidden-lexicon scan wired into verification; humanized audit/activity labels | REDG-03 automated scan green across pages | Language system enforced |
| **R5** | States, responsive, a11y hardening | Page-state coverage; overflow assertions; focus-path check; touch targets | State screenshot matrix; REDG-06/07 pass | REDESIGN_IMPLEMENTED_LOCAL |
| **R6** | Design-fidelity gate + owner retry | Reference-vs-implementation checklist per surface; fresh full screenshot pack; VS4-H01 review package; amended Plane-1 rows re-anchored to `/review` (D3) | Owner review record filled (`APPROVE` / `APPROVE_WITH_EXCEPTIONS` / `REJECT`) | VS4-H01 outcome as recorded — nothing stronger |

Sequencing rationale: R1 gives the owner a directional read on the highest-risk surface within one increment instead of after a full rebuild; R6's fidelity gate becomes a permanent pre-human checkpoint so structural PASS can never again mask design failure (fixes the report's Verification Gap).

---

## 10. Verification and Acceptance

### 10.1 Redesign gates (mapped to DS-S scenarios and VS4-H01 retry criteria)

| ID | Gate | Mode | Maps to |
| --- | --- | --- | --- |
| REDG-01 | Home first viewport shows shell, headline, drop, ask, suggested prompts (desktop) in reference priority; recents/states/next-steps/rail present on page | AUTOMATED (region markers + screenshot) + HUMAN | DS-S01; retry 1, 3, 5 |
| REDG-02 | No review/verification/package content on any product surface; `/review` contains it all | AUTOMATED (absence + presence scan) | Retry 4 |
| REDG-03 | Forbidden lexicon absent from product surfaces; chips use plain-language labels | AUTOMATED (lexicon scan §5.3) | Retry 4; contract §9 |
| REDG-04 | Every product record page reachable by real route; nav contains exactly the §3.2 model | AUTOMATED (route + nav check) | Retry 1 |
| REDG-05 | Citation disclosure ladder works on a real Brief (chip→popover→drawer→audit) | AUTOMATED (DOM) + HUMAN | DS-S05/S07 analog; VS5 CS-VAL alignment |
| REDG-06 | Page-state coverage incl. day-zero empty states, no fabricated data | AUTOMATED (state matrix) + HUMAN spot-check | §5.4, §6.4 |
| REDG-07 | Mobile keeps first-value hierarchy; zero body-level horizontal overflow on all pages | AUTOMATED (CDP assertion + screenshots) | Retry 7 |
| REDG-08 | Rendered styles derive from tokens; no out-of-pipeline hex/px | AUTOMATED (lint + computed-style sample) | Contract §5 |
| REDG-09 | Fresh desktop+mobile screenshot pack attached to review package | AUTOMATED generation, HUMAN consumed | Retry 8 |
| REDG-10 | Owner walkthrough decision recorded | HUMAN_REQUIRED | Retry 9; VS4-H01 |

### 10.2 What automation proves — and does not

Automated gates prove layout regions exist, internal language is absent, tokens are wired, and hierarchy survives viewports. They do not prove the product *feels* calm or that a stranger trusts a Brief — REDG-10 (owner) and later VS5 external sessions own those. No automated result may be reported as human acceptance (existing repo rule, unchanged).

### 10.3 Verdict ladder for this effort

`DESIGN_GUIDANCE_ADOPTED` (owner accepts this document) → `REDESIGN_IMPLEMENTED_LOCAL` (R0–R5 + REDG-01..09 green) → `VS4_H01_ACCEPTED` (owner record) → VS5 external entry unblocked. Each verdict claims nothing beyond its evidence; VS5 value claims remain governed by CS-VAL.

---

## 11. Risks and Open Decisions

**Decisions needed from owner:**

| ID | Decision | Recommendation |
| --- | --- | --- |
| D1 | Brief detail direction (§4.2) has no reference image | Approve §4.2 composition as the reference-equivalent for the Brief surface, or supply a reference image before R2 |
| D2 | Inbox placement | Secondary nav now; revisit at VS6 |
| D3 | Existing VS4 Plane-1 rows assert markers on the monolithic Home; eviction re-anchors them to `/review` | Amend affected row expectations inside the recovery slice (no new contracts; recorded as a dated change) |
| D4 | HTML route prefix vs. existing JSON routes | Serve HTML at plain paths, keep JSON API as-is; confirm at R0 |
| D5 | UI copy language | Keep English for VS5 external testing |

**Risks:**

- **Fabricated-data temptation** while matching references — mitigated by §5.4 rules and REDG-06's no-fabrication check.
- **Marker/layout re-coupling** — mitigated by §8.5 (markers are metadata) and REDG-02/03 scans.
- **Scope creep into dormant surfaces** (admin, vendor explorer, graph) — mitigated by §4.10 and the ADR-0007 freeze.
- **Verification-apparatus growth** — this plan adds absence/overflow checks to the *existing* harness rather than new report families, staying inside ADR-0007 §6's cap.
- **Timeline pressure against VS5** — R1 gives the owner an early directional checkpoint; if R1 misses the mark, the direction can be corrected before R2–R6 spend.

---

## 12. Source Register

| Source | Used for |
| --- | --- |
| `docs/verification-reports/VS4_H01_UI_UX_REJECTION_REPORT_2026-07-04.md` | Rejection record, gap analysis, recovery phases, retry criteria |
| `reports/human-gates/vs4/filled-records/VS4-H01.review-record.json` | Owner decision (REJECT) |
| `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`, `DESIGN_CONCEPT_SYSTEM_V0_3.md`, `tokens/cornerstone_design_tokens_v0_3.json` | Doctrine, surfaces, tokens, redlines, DS-S scenarios |
| `docs/design/reference-images/*` (8 images) | Target layouts (structure only; §5.4) |
| `packages/cornerstone_cli/product_runtime.py` | Verified current implementation (monolith, anchors, palette, markers, routes) |
| `reports/browser/vs4-product-alpha-ui-daily-loop-slice-001{,-mobile}/home.png` | Verified current visual state, desktop + mobile |
| `docs/adr/ADR-0007-product-value-first-reset.md`, `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` | Spine, scope freeze, dormancy, CS-VAL alignment, VS5 gate |
