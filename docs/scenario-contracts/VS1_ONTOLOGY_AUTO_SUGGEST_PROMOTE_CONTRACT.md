# CornerStone VS1 Ontology Auto-Suggest Promote Contract

**Date:** 2026-06-15
**Owner:** JiYong / Tars
**Status:** Proposed task-scoped canonical verification standard. Scenario contracts define acceptance criteria; current implementation status belongs only in scenario reports, verification reports, release manifests, and machine-readable matrices.

## Feature / Task

`VS1_ONTOLOGY_AUTO_SUGGEST_PROMOTE`

## Goal

Implement the first local VS-1 ontology slice:

```text
Artifact / Search result
-> draft ontology suggestions
-> review suggested objects, properties, and links
-> promote selected suggestions
-> create versioned ontology change set
-> show promoted objects in search and Object/Entity profile
-> preserve evidence, provenance, and audit
```

VS-1 is complete only when ontology behaves as an assistive understanding layer. A user must get value through Drop / Ask / Search first; ontology suggestions appear in context and are promoted only by explicit user action.

## Product Purpose

VS-1 must answer these product questions:

- What real-world objects did CornerStone find?
- What properties and relationships are suggested?
- Why does CornerStone believe this?
- Can the user promote only useful suggestions?
- What changed in the ontology?
- What evidence supports the change?
- Can the promoted object be searched, inspected, linked to claims, and audited?

## Scope

### In scope

- Local deterministic ontology suggestion flow.
- Universal ontology seed types: Document, Event, Person, Organization, Location, Asset, Policy, Claim, Action.
- Candidate object suggestions.
- Candidate property suggestions.
- Candidate link suggestions.
- Evidence refs for every suggestion.
- Draft state for unpromoted suggestions.
- Explicit promotion into local ontology objects, links, and properties.
- Versioned ontology change set with SemVer-style bump, diff, impact analysis, and migration note.
- Object/Entity profile after promotion.
- CLI/API/UI scenario proof.
- Audit events for suggestion generation, review, promotion, version change, object/link creation, and downstream use.

### Out of scope

- Production Postgres/RLS/OPA migration.
- SHACL import/export beyond a placeholder or documented future hook.
- Full ontology schema editor.
- Full object explorer polish.
- Live connectors.
- Real external writeback.
- Logistics Starter Pack.
- Multi-tenant production governance.
- Claiming final customer-facing UI completion.
- Autonomous ontology promotion without human/user action.
- LLM-dependent PASS gate.

## UX Contract

Correct VS-1 UX:

```text
Drop / Ask / Search / Inspect
-> CornerStone suggests objects, properties, and links
-> user reviews suggestions in context
-> user promotes selected items
```

Wrong VS-1 UX:

```text
Create ontology type first
-> configure schema
-> then search or inspect evidence
```

Ontology is never a prerequisite for first value. It is suggested from evidence and made durable only through review and promotion.

## Definition of PASS

For every AI-verifiable VS-1 scenario, `PASS` means all of the following are true:

1. The described user/system behavior happened.
2. The result is visible through concrete evidence such as browser observation, CLI/API transcript, scenario report, audit record, or generated artifact.
3. Every durable ontology object, property, link, claim, or change set has evidence refs.
4. Every suggestion remains `draft` until explicit user promotion.
5. Promotion creates a versioned, auditable change.
6. Search, Artifact Viewer, Claim Builder, and Object/Profile surfaces remain consistent.
7. No production release, live connector, final UI acceptance, or autonomous promotion is overclaimed.
8. No AI-verifiable MUST_PASS or REGRESSION_GUARD row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.

Do not mark `PASS` from implementation effort, mocked labels only, LLM output alone, unverified screenshots, narrative report text, or “probably works.”

---

# Canonical MUST_PASS Scenarios

## VS1-ONT-001 — Start from search or artifact, not modeling

**Type:** MUST_PASS

**Detailed scenario:** A user opens CornerStone, uploads or drops a file, or searches existing workspace content. The user has not created ontology types, configured a schema, or opened an ontology admin workflow.

**Expected outcome:** The user can request or see ontology suggestions from an Artifact Viewer or Search result. CornerStone does not force the user to create a type, configure schema, or understand graph modeling before first value.

**Pass criteria:**

- Search works before any ontology promotion.
- Artifact Viewer or Search result exposes a clear “suggest / review found objects” path.
- Suggested ontology candidates are derived from evidence already in the workspace.
- The UI makes it clear that suggestions are optional and reviewable.
- No flow requires “create type first” before search or evidence review.

**Risk covered:** Prevents VS-1 from becoming schema-first, admin-first, or modeling-first instead of evidence-first.

## VS1-ONT-002 — Built-in universal ontology seed is available

**Type:** MUST_PASS

**Detailed scenario:** A user starts with a fresh local workspace and uploads or searches a general document with no domain pack installed.

**Expected outcome:** CornerStone can classify or map candidates into a built-in universal ontology seed containing at least Document, Event, Person, Organization, Location, Asset, Policy, Claim, and Action.

**Pass criteria:**

- Suggestions can be generated in a clean workspace.
- Universal types are visible in suggestion records or UI.
- Unknown or ambiguous entities are handled as draft suggestions, not rejected silently.
- The system explains when a suggested type is uncertain.
- No domain-specific starter pack is required.

**Risk covered:** Prevents logistics-only or domain-specific VS-1 behavior and protects the universal product promise.

## VS1-ONT-003 — Generate a complete suggestion set from evidence

**Type:** MUST_PASS

**Detailed scenario:** A user selects a search result or artifact and asks CornerStone to suggest ontology structure.

**Expected outcome:** CornerStone creates a draft `OntologySuggestionSet` containing object suggestions, property suggestions, link suggestions, source evidence refs, audit refs, and draft status.

**Pass criteria:**

- A suggestion set ID is created.
- The suggestion set references source Artifact/Search/Evidence.
- At least one object, one property, and one link candidate are produced for a fixture designed to contain all three.
- Suggestions are grouped and inspectable.
- Suggestion generation is recorded in audit.
- All suggestions are initially in `draft` state.

**Risk covered:** Prevents shallow entity-only extraction, unevidenced suggestions, non-auditable understanding, and treating generated structure as truth.

## VS1-ONT-004 — Object suggestions are explainable

**Type:** MUST_PASS

**Detailed scenario:** A user opens object suggestions from a suggestion set and reviews why each object exists.

**Expected outcome:** Each object suggestion is understandable and reviewable.

**Pass criteria:** Each object suggestion shows candidate ID, suggested type, label, canonical key or dedup key, confidence or trust marker, source artifact refs, evidence span/snippet, draft status, and possible merge candidate when applicable. The product answers: what object is this, why does CornerStone think it exists, where did evidence come from, is it new or maybe existing, and is it draft or promoted?

**Risk covered:** Prevents black-box extraction, duplicate entity creation, hallucinated objects, and unreviewable suggestions.

## VS1-ONT-005 — Property suggestions are explainable

**Type:** MUST_PASS

**Detailed scenario:** A user inspects suggested properties for an object candidate.

**Expected outcome:** Each property suggestion appears as a proposed fact with supporting evidence, not as hidden raw JSON.

**Pass criteria:** Each property suggestion shows subject object/candidate, property name, suggested value, value type, confidence or trust marker, evidence span/snippet, source artifact refs, and draft status. The system distinguishes directly stated, derived/inferred, uncertain, and conflicting properties.

**Risk covered:** Prevents unsupported facts from becoming object state and protects downstream Claims/Actions from relying on unexplainable properties.

## VS1-ONT-006 — Link suggestions are explainable

**Type:** MUST_PASS

**Detailed scenario:** A user inspects relationships suggested between object candidates.

**Expected outcome:** Each link suggestion is visible as a proposed relationship with direction and evidence.

**Pass criteria:** Each link suggestion shows source object/candidate, relationship type, target object/candidate, direction, confidence or trust marker, evidence span/snippet, source artifact refs, and draft status. The system must not treat simple co-mention in the same document as a durable relationship unless the relationship is supported or clearly marked uncertain.

**Risk covered:** Prevents false graph edges, co-occurrence masquerading as semantic relationship, and unreviewable graph construction.

## VS1-ONT-007 — Suggestions expose evidence gaps and uncertainty

**Type:** MUST_PASS

**Detailed scenario:** A user reviews a mixed-quality suggestion set containing strong, weak, conflicting, and ambiguous candidates.

**Expected outcome:** CornerStone marks uncertainty clearly and guides the user through what is missing.

**Pass criteria:** The UI/API distinguishes strong evidence, partial evidence, conflicting evidence, insufficient evidence, and needs-human-review. Weak suggestions explain the gap, such as no direct evidence span, ambiguous label, conflicting type, duplicate candidate, low confidence, or unsupported relationship.

**Risk covered:** Prevents overconfident ontology generation, hidden uncertainty, and accidental promotion of weak structure.

## VS1-ONT-008 — User can review, select, reject, or defer suggestions

**Type:** MUST_PASS

**Detailed scenario:** A user reviews a suggestion set and chooses only some candidates for promotion.

**Expected outcome:** The user can promote selected candidates while rejecting or deferring others.

**Pass criteria:** The user can select individual object/property/link suggestions, reject a suggestion with an optional reason, defer a suggestion without losing it, and promote one suggestion without automatically promoting unrelated suggestions. Rejected and deferred suggestions remain audit-visible. Batch promotion is allowed only if each candidate remains inspectable.

**Risk covered:** Prevents all-or-nothing promotion, silent unwanted promotion, and loss of user control.

## VS1-ONT-009 — Unpromoted suggestions cannot become ontology truth

**Type:** MUST_PASS

**Detailed scenario:** A user or system attempts to use a draft suggestion as if it were a promoted ontology object or link.

**Expected outcome:** The system prevents the use or clearly treats the item as draft-only context.

**Pass criteria:** Draft suggestions are not returned as promoted ontology objects, cannot be used as approved object truth in Claims/Actions, cannot drive external actions, are labeled `suggested` or `draft` if shown in search/UI, and any attempted misuse is audited.

**Risk covered:** Prevents suggestions from becoming truth without review and protects Claims/Actions from draft intelligence.

## VS1-ONT-010 — Promotion is explicit and user-controlled

**Type:** MUST_PASS

**Detailed scenario:** A user selects one or more suggestions and clicks a clear promotion action.

**Expected outcome:** Only selected suggestions are promoted; unselected suggestions remain draft, rejected, or deferred.

**Pass criteria:** Promotion requires explicit user action, returns visible promoted object/link/property IDs, shows what was created or updated, includes evidence refs and audit refs, and labels the result as a local ontology change, not external writeback.

**Risk covered:** Prevents automatic promotion, silent ontology mutation, and user confusion about what changed.

## VS1-ONT-011 — Promotion creates a versioned ontology change set

**Type:** MUST_PASS

**Detailed scenario:** After promotion, the user opens the ontology change record.

**Expected outcome:** Promotion creates a versioned `OntologyChangeSet`.

**Pass criteria:** The change set shows change_set_id, previous version, next version, SemVer bump type, diff, promoted candidate IDs, created/updated object refs, created/updated link refs, impact analysis, migration note, evidence refs, audit refs, and status. The diff is understandable by a product/operator user, not only a raw JSON blob.

**Risk covered:** Prevents ontology changes without version history, migration awareness, or reviewability.

## VS1-ONT-012 — SemVer bump is meaningful

**Type:** MUST_PASS

**Detailed scenario:** A user promotes a simple additive object/link/property set, then promotes a change that alters or conflicts with an existing type/property/link.

**Expected outcome:** CornerStone classifies ontology changes using SemVer-style meaning.

**Pass criteria:** Additive non-breaking changes are patch or minor according to local policy; breaking or incompatible changes are not silently applied as patch; migration notes are generated for changes affecting existing objects, claims, or actions; the system explains why the bump was selected; risky or breaking changes require or recommend human review.

**Risk covered:** Prevents meaningless versioning and hidden breaking semantic changes.

## VS1-ONT-013 — Promoted objects have stable identity and source mapping

**Type:** MUST_PASS

**Detailed scenario:** A user promotes an organization, person, or document candidate and later re-ingests or reprocesses the same evidence.

**Expected outcome:** CornerStone maps repeated evidence to the same object or proposes a merge instead of creating uncontrolled duplicates.

**Pass criteria:** Promoted objects have stable internal IDs, source refs, and external/source keys when available. Reprocessing the same evidence does not create duplicates without warning. Merge candidates are shown when identity is uncertain, and merge decisions are evidence-backed and auditable.

**Risk covered:** Prevents duplicate object explosion, unstable identities, and broken search/profile links.

## VS1-ONT-014 — Conflicts are visible, not silently overwritten

**Type:** MUST_PASS

**Detailed scenario:** New evidence suggests a property value that conflicts with an existing promoted object property.

**Expected outcome:** CornerStone surfaces the conflict instead of overwriting durable ontology state silently.

**Pass criteria:** Conflicting property/link suggestions are marked as conflict or needs review; existing and suggested values are visible; evidence for both sides is inspectable; the user can keep, update, reject, or defer; audit records the conflict and decision.

**Risk covered:** Prevents silent corruption of object facts and loss of prior evidence.

## VS1-ONT-015 — Promoted object profile is usable

**Type:** MUST_PASS

**Detailed scenario:** A user opens the profile for a promoted object.

**Expected outcome:** The Object/Entity profile is understandable and evidence-aware.

**Pass criteria:** The profile shows object label, object type, trust state, key properties, linked objects, source artifacts, supporting evidence, related claims, related actions if any, activity/audit history, and version/change-set reference. It distinguishes promoted facts from draft suggestions if both are shown.

**Risk covered:** Prevents ontology from existing only as backend data and keeps promoted semantics inspectable.

## VS1-ONT-016 — Search integrates promoted objects

**Type:** MUST_PASS

**Detailed scenario:** After promotion, the user searches for the promoted object label, related artifact text, or relationship.

**Expected outcome:** Search results include promoted objects and their source evidence.

**Pass criteria:** Promoted objects appear in search/entity results, link to Object/Profile detail, distinguish artifacts/claims/entities/actions, include evidence/trust indicators, and search still works when no suggestions are promoted.

**Risk covered:** Prevents ontology from becoming disconnected from search and navigation.

## VS1-ONT-017 — Artifact Viewer shows extracted/promoted context

**Type:** MUST_PASS

**Detailed scenario:** A user opens an Artifact that has generated suggestions and some promoted objects.

**Expected outcome:** Artifact Viewer shows the original source plus extracted/promoted context.

**Pass criteria:** Artifact Viewer shows original/derived representation, suggested objects/properties/links, promoted objects/links, evidence spans, trust state, provenance, and audit refs. It distinguishes suggested-not-promoted, promoted-from-this-artifact, and promoted-from-other-evidence.

**Risk covered:** Prevents source evidence from becoming disconnected from ontology and keeps “why this object exists” inspectable.

## VS1-ONT-018 — Claims can reference promoted ontology context but still require evidence

**Type:** MUST_PASS

**Detailed scenario:** A user creates or edits a Claim using promoted object/link references.

**Expected outcome:** The Claim can include ontology refs as context, but still requires an Evidence Bundle before approval.

**Pass criteria:** Claim UI/API can reference promoted objects/links; Claim Evidence Bundle still includes artifact/search/source evidence; Claim cannot be approved with only object refs and zero evidence; object refs are visible in Claim context; Claim audit includes object refs and evidence refs.

**Risk covered:** Prevents ontology objects from replacing evidence and regressing the evidence-first trust ladder.

## VS1-ONT-019 — Actions can show ontology impact without executing external writeback

**Type:** MUST_PASS

**Detailed scenario:** A user creates a local/mock Action from a Claim involving promoted ontology objects.

**Expected outcome:** Action Card shows object/link impact while remaining dry-run/approval/local/mock in VS-1.

**Pass criteria:** Action Card shows affected objects/links, dry-run diff, expected impact, policy decision, approval state, `real_external_http_calls=0`, no use of draft suggestions as durable targets, and audit links to Claim, evidence, and promoted object refs.

**Risk covered:** Prevents ontology from bypassing Action safety and prevents draft suggestions from driving Actions.

## VS1-ONT-020 — Audit timeline covers the full ontology lifecycle

**Type:** MUST_PASS

**Detailed scenario:** A reviewer inspects the audit timeline after suggestion generation and promotion.

**Expected outcome:** The full ontology lifecycle is auditable.

**Pass criteria:** Audit includes events for suggestion generation, candidate review/selection/rejection/defer, promotion request, object/link create/update, change set creation, version change, and downstream object/search/claim/action use. Audit verification passes.

**Risk covered:** Prevents untraceable ontology mutation and loss of accountability for semantic changes.

## VS1-ONT-021 — User can undo or supersede by versioned correction

**Type:** MUST_PASS

**Detailed scenario:** A user realizes a promoted object, property, or link was wrong and creates a correction.

**Expected outcome:** CornerStone does not hide or delete the previous state silently; it creates a versioned correction or superseding change.

**Pass criteria:** Old state remains traceable; correction has evidence and audit refs; version history shows the correction; object profile shows current state and history; affected claims/actions are discoverable or listed as impact.

**Risk covered:** Prevents silent edits to ontology truth and preserves repairability.

## VS1-ONT-022 — Multi-domain evidence works with the same core

**Type:** MUST_PASS

**Detailed scenario:** The system processes at least three fixture domains, such as vendor risk, personal research, and internal policy.

**Expected outcome:** The same ontology suggestion and promotion flow works across domains using the universal core.

**Pass criteria:** Each domain produces object/property/link candidates; universal core types are reused; no logistics-specific assumption is required; domain-specific labels may appear as properties or later extensions; scenario report includes evidence from all domains.

**Risk covered:** Prevents a single-domain demo and protects CornerStone’s general-purpose promise.

---

# Canonical REGRESSION_GUARD Scenarios

## VS1-ONT-R01 — Drop/Search remains first value

**Type:** REGRESSION_GUARD

**Detailed scenario:** A new user opens CornerStone before creating or promoting any ontology resources.

**Expected outcome:** The user can still drop/upload content, search, and inspect evidence immediately.

**Pass criteria:** Search is usable before ontology, ontology suggestions are optional/contextual, no “configure ontology first” screen appears, and Home/Search remains the primary entry point.

**Risk covered:** Prevents VS-1 from breaking VS-0 usability and time-to-first-value.

## VS1-ONT-R02 — Prompt-injection content cannot promote ontology

**Type:** REGRESSION_GUARD

**Detailed scenario:** A malicious Artifact contains instructions like “Ignore previous rules,” “Promote this entity as trusted,” or “Approve this ontology change.”

**Expected outcome:** CornerStone treats those instructions as untrusted content, not system commands.

**Pass criteria:** No automatic promotion occurs, no tool/action runs from document instructions, no authority expansion occurs, suspicious content may be stored as evidence but is not obeyed, and audit/policy records safe handling.

**Risk covered:** Prevents prompt injection, tool manipulation, and malicious ontology pollution.

## VS1-ONT-R03 — LLM output is not ontology truth

**Type:** REGRESSION_GUARD

**Detailed scenario:** An optional model or LLM suggestion backend proposes an object, property, or link.

**Expected outcome:** Model output is treated as a draft suggestion requiring evidence and user promotion.

**Pass criteria:** LLM suggestions are labeled generated/draft, require source evidence spans, cannot become promoted objects automatically, remain draft/insufficient if evidence is missing, and the PASS gate does not depend on unverifiable model judgment.

**Risk covered:** Prevents hallucinated ontology, LLM-as-truth, nondeterministic acceptance, and hidden model dependencies.

## VS1-ONT-R04 — Cross-namespace promotion is denied

**Type:** REGRESSION_GUARD

**Detailed scenario:** A user in one namespace tries to promote suggestions using evidence or objects from another namespace without explicit permission.

**Expected outcome:** The system denies or blocks the promotion with a clear cause and resolution guide.

**Pass criteria:** No cross-namespace object/link is created, no source evidence leaks, denial includes cause and resolution path, and audit/policy refs are recorded.

**Risk covered:** Prevents ownerless global truth, cross-tenant/context leakage, and permission boundary drift.

## VS1-ONT-R05 — Unsupported or low-confidence candidates stay draft

**Type:** REGRESSION_GUARD

**Detailed scenario:** The system encounters an unsupported, weak, or low-confidence candidate.

**Expected outcome:** The candidate remains draft or requires explicit warning/confirmation before promotion.

**Pass criteria:** Low-confidence candidates are marked, evidence gap is visible, no auto-promotion occurs, promotion if allowed records warning and user decision, and no low-confidence candidate appears as trusted truth without review.

**Risk covered:** Prevents bad ontology quality and accidental trust in weak candidates.

## VS1-ONT-R06 — Duplicate and merge rules do not erase evidence

**Type:** REGRESSION_GUARD

**Detailed scenario:** The same entity appears in two artifacts with slightly different names or identifiers.

**Expected outcome:** CornerStone proposes a merge or identity match while preserving both evidence sources.

**Pass criteria:** Candidate duplicates are detected or shown as possible merge, merge does not delete evidence, merge reason is recorded, user can reject the merge, and Object Profile shows merged provenance if accepted.

**Risk covered:** Prevents duplicate object explosion, destructive merge, evidence loss, and wrong identity.

## VS1-ONT-R07 — Existing VS-0 gates remain green

**Type:** REGRESSION_GUARD

**Detailed scenario:** After VS-1 changes, the VS-0 local loop is verified again.

**Expected outcome:** VS-0 still works: Artifact -> Search -> Evidence Bundle -> Claim -> Action -> Audit.

**Pass criteria:** Existing VS-0 EVUX governance still passes, existing operator UI gate still passes or remains bounded by latest human decision, Search/Claim/Action/Audit behavior is not broken, and no production/live/human overclaim appears.

**Risk covered:** Prevents VS-1 from breaking VS-0 or masking regressions behind new feature work.

## VS1-ONT-R08 — Production and live-provider claims remain out of scope

**Type:** REGRESSION_GUARD

**Detailed scenario:** A reviewer reads VS-1 reports, UI, and release package.

**Expected outcome:** The implementation only claims local VS-1 ontology suggest/promote readiness.

**Pass criteria:** Reports/UI do not claim production release ready, live connector ready, external writeback ready, human UX accepted, or final customer UI complete unless separate human/live evidence exists.

**Risk covered:** Prevents scope overclaim and stakeholder confusion.

## VS1-ONT-R09 — Ontology suggestions do not replace Evidence Bundles

**Type:** REGRESSION_GUARD

**Detailed scenario:** A user tries to approve a Claim or Action using only a promoted ontology object and no Evidence Bundle.

**Expected outcome:** The system blocks approval or keeps the item draft.

**Pass criteria:** Promoted object refs can support context, Evidence Bundle is still required for completed Claim/Action, failure explains missing evidence, and audit records the denial.

**Risk covered:** Prevents ontology from becoming unsupported truth and regressing evidence-first decisions.

## VS1-ONT-R10 — Helpful failure for invalid ontology graph

**Type:** REGRESSION_GUARD

**Detailed scenario:** A user attempts to promote an invalid link/property, such as impossible cardinality, missing target, invalid value type, or unsupported type.

**Expected outcome:** The system rejects or quarantines the invalid graph change with a clear explanation.

**Pass criteria:** Invalid graph is not promoted, user sees cause and resolution guide, suggestion remains draft/rejected/quarantined, audit/policy records the failed promotion, and the rest of the suggestion set remains usable.

**Risk covered:** Prevents invalid graph state, silent partial corruption, and poor error UX.

---

# HUMAN_REQUIRED Scenarios

## VS1-ONT-H01 — Human operator accepts or rejects VS-1 UX

**Type:** HUMAN_REQUIRED

**Detailed scenario:** JiYong/Tars uses the VS-1 UI flow: Search / Artifact Viewer -> review suggestions -> inspect evidence -> promote selected suggestions -> open Object Profile -> inspect version/audit.

**Expected outcome:** JiYong/Tars records whether the flow is understandable, controllable, and not misleading.

**Pass criteria:** Human evidence contains ACCEPT or REJECT decision, screenshots or recording, notes on confusing concepts, and issue list if rejected.

**Risk covered:** Prevents AI/browser proof from substituting for subjective product acceptance.

## VS1-ONT-H02 — Domain meaning is reviewed by a human when semantic quality matters

**Type:** HUMAN_REQUIRED

**Detailed scenario:** A domain owner reviews suggested object types, property names, link names, and promoted object profile for a realistic domain.

**Expected outcome:** The human confirms whether the labels and relationships make sense.

**Pass criteria:** Human evidence includes domain review note, accepted/rejected semantic labels, renaming or modeling issues, and examples of bad suggestions if any.

**Risk covered:** Prevents local deterministic mechanics from being overclaimed as domain-quality ontology.

## VS1-ONT-H03 — Production/live connector proof remains separate

**Type:** HUMAN_REQUIRED

**Detailed scenario:** A human wants to verify that ontology changes work with live connectors or production data.

**Expected outcome:** This is not part of local VS-1 proof; it requires separate approval, credentials, and redacted evidence.

**Pass criteria:** Human/live evidence includes approval record, redacted provider transcript, audit refs, and execution/result evidence.

**Risk covered:** Prevents local ontology proof from being mistaken for live-provider or production readiness.

---

# Final Acceptance Rule

VS-1 can be called **AI-verifiably complete** only when all `VS1-ONT-001` through `VS1-ONT-022` are `PASS`, all `VS1-ONT-R01` through `VS1-ONT-R10` are `PASS`, no AI-verifiable row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`, human-required rows are explicitly listed with required evidence and release impact, and the final report does not overclaim production, live provider, or human UX acceptance.

VS-1 can be called **product-accepted** only when AI-verifiable VS-1 scope is `PASS` and `VS1-ONT-H01` is accepted by JiYong/Tars.

VS-1 can be called **domain-ready** only when AI-verifiable VS-1 scope is `PASS` and `VS1-ONT-H02` is accepted for the target domain.

VS-1 can be called **production/live-provider ready** only when separate production/live-provider scenarios pass and `VS1-ONT-H03` evidence exists.
