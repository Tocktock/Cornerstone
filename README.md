# CornerStone

**Date:** 2026-07-04
**Owner:** JiYong / Tars
**Status:** Product-value-first reset (see `docs/adr/ADR-0007-product-value-first-reset.md`). Local structural substrate is real and verified; the intelligence layer is the active build; next proof point is external.
**Canonical spelling:** Use **CornerStone** for product/project text.

## What CornerStone Is Becoming

**Drop messy input, get a brief you can defend — every load-bearing statement traceable to its source, every decision recorded with an audit trail.**

CornerStone is being built for people who must stand behind their conclusions: operators, analysts, founders, and eventually teams. Generic AI tools produce confident text nobody can verify. CornerStone's promise is *briefs with receipts* — and an evidence substrate underneath (immutable originals, content hashes, trust states, tamper-evident audit) that makes "says who?" a one-click question.

The active product spine is:

```text
Drop / Ask -> Evidence-backed Brief -> Decision -> Audit
```

Everything on the spine is active. Everything off the spine is dormant until user evidence pulls it back (see Dormant Systems below). The long-term direction — living memory, governed actions, learning loops — is preserved in `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`, explicitly labeled as future-facing.

## Current State, Honestly

Reviewed and live-tested 2026-07-04 (`docs/adr/ADR-0007-product-value-first-reset.md`).

**Verified working (Plane 1 — structural):**

- Immutable artifact store with SHA256 content addressing, dedupe, and forced-untrusted user input.
- Hash-chained, tamper-evident audit ledger (`cornerstone audit verify`).
- Evidence bundles linking briefs/claims to source artifacts; owner/namespace scoping on every record.
- Local runtime with a calm web UI, full CLI parity (`--json`, evidence refs, audit refs), and real Chrome-CDP browser proofs.
- Deterministic scenario verification harness across VS0–VS4 (27/28 VS4 rows structurally green; `VS4-H01` owner review still open).

**Not yet real (the active build):**

- **There is no model integration yet.** Briefs are currently extractive snippets of the user's own input plus fixed strings; Ask returns a canned deferral sentence. The Understand/Decide stages of the loop do not exist yet.
- Four product-value scenarios are recorded as open **FAIL** against the current behavior (echo briefs, boilerplate uncertainty, unearned trust labels, non-answer Ask) in `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`. They flip only with VS5 evidence.

**Never claimed:** production readiness, live provider execution, real tenancy/security posture, on-prem readiness, external-user validation. No external user has used CornerStone yet; changing that is the point of the current milestone.

## The Next Proof Point

Not more internal PASS counts. The next meaningful proof is:

> **An unfamiliar external user, on their own real messy input, gets a brief they understand and trust — every load-bearing statement one click from its source — within 10 minutes.**

This is the VS5 stranger test (`VS5-EXT-001/002`), with a 3-minute unedited session recording as the deliverable demo artifact.

## Milestones

| Milestone | Focus | Verdict earned / targeted |
|---|---|---|
| VS0–VS4 (closed) | Structural substrate: artifacts, evidence, audit, policy records, UI shell, daily-loop skeleton, verification harness | `STRUCTURAL_READY` (strongest claim these can ever support; `VS4-H01` owner review still open) |
| **VS5 (active)** | **Citation-grounded, model-backed Brief and Ask; earned trust labels; eval corpus; external stranger test** | targets `VALUE_VERIFIED_EXTERNAL` — `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md` |
| VS6 (next) | Daily loop: one read-only ingest source, self-filling inbox, morning digest, retrieval at volume, ~20 external users | `docs/scenario-contracts/VS6_DAILY_LOOP_CONTRACT.md` |
| VS7 (then) | Wedge validation: design partners, willingness-to-pay evidence, keep/kill on off-spine surfaces, dormancy dispositions | `docs/scenario-contracts/VS7_WEDGE_VALIDATION_CONTRACT.md` |

Milestone sequencing rationale and the scope freeze (no new scenario contracts, report families, trace counters, or CLI command families off the spine until VS5 closes) are in ADR-0007.

## Two Verification Planes

Structural PASS counts can no longer support product-value claims. Authority: `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`.

- **Plane 1 — Structural:** deterministic validators over records, boundaries, labels, audit chains, CLI transcripts. Judge: code. Supports at most `STRUCTURAL_READY`.
- **Plane 2 — Product value:** grounding, zero fabricated citations, faithfulness, synthesis-beyond-extraction, honest uncertainty, earned trust labels, direct answers, external comprehension and trust (CS-VAL-001..010). Judges: deterministic citation-integrity checks + humans; local LLM judges are advisory only. Supports `VALUE_VERIFIED_LOCAL` / `VALUE_VERIFIED_EXTERNAL`.

**Model assumptions (local-first):** generation `ornith:35b`, embeddings `qwen3-embedding:0.6b`, both via Ollama (verified installed). The deterministic `local_test` provider remains the Plane 1 CI baseline. External model providers are optional and future-facing, named per-scenario when assumed.

## Dormant Systems (honest register)

These exist in the repo with real code, frozen contracts, and prior structural evidence. They are **not part of the current product story** and must not appear in active roadmaps, product claims, or default UX. Strategic possibility preserved; reactivation requires a dated user-evidence rationale (disposition review lands in VS7).

| System | State | Where it lives |
|---|---|---|
| ConnectorHub expansion (providers, human-gate apparatus) | DORMANT | `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md` |
| VS2 policy/tenancy/egress (Postgres RLS, OPA, egress topology) | DORMANT (harness exists; not wired into product runtime) | `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md` |
| VS3 on-prem security / trusted extension substrate | DORMANT (frozen, never started) | `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md` |
| VS1 ontology suggest/promote | DORMANT (structurally closed; off the spine) | `docs/scenario-contracts/VS1_ONTOLOGY_AUTO_SUGGEST_PROMOTE_CONTRACT.md` |
| Brain routing / judge / model ledger; agent orchestration / packs; autopilot / missions; capsules / decision-cards; memory promotion machinery | DORMANT (CLI stubs + fixtures only; no real capability) | `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` §2.4 |

Action Cards and Memory/Wiki candidates remain visible in the UI as review drafts, but are not part of the active value claim.

## Run It Locally

```sh
export PATH="$PWD:$PATH"
cornerstone ready --json        # local_scenario_ready=true, production_release_ready=false
cornerstone runtime serve --port 8787
```

Then open the local UI, or drive the same loop from the CLI:

```sh
cornerstone artifact ingest --text "paste anything messy here" --source user_paste --json
cornerstone search query "your topic" --json
cornerstone evidence bundle create --search-snapshot-id <id> --json
cornerstone brief create --evidence-bundle-id <id> --json
cornerstone claim create --evidence-bundle-id <id> --statement "..." --json
```

**What you will see today:** the full evidence/audit loop working structurally — and a brief that is still extractive (your own text back, labeled and linked). That gap is VS5. Do not demo this as an intelligent product yet.

## VS0 Runtime Acceptance Quickstart

Executable active-spine acceptance quickstart (isolated fixture state, local deterministic fallback, zero external HTTP calls):

```sh
cornerstone quickstart verify vs0-runtime-acceptance --json
```

The quickstart writes `tmp/quickstart/vs0-runtime-acceptance-current.json` and runs Artifact ingest → Search → Evidence Bundle → honest fallback Brief → draft Claim → Audit verification. It does not approve claims, create actions, invoke an external model, or claim product value.

Historical broader local acceptance walkthrough (fixture data and mocked actions), kept verbatim-compatible for the structural verifiers:

```sh
cornerstone runtime serve --port 8787
cornerstone artifact ingest fixtures/vs0/packs/01_artifact_basic/input.txt --state-dir data/quickstart-vs0 --json
cornerstone search query alpha-evidence-anchor --state-dir data/quickstart-vs0 --json
cornerstone evidence bundle create --search-snapshot-id <search_snapshot_id> --state-dir data/quickstart-vs0 --json
cornerstone claim create --evidence-bundle-id <evidence_bundle_id> --statement "The Alpha evidence anchor is ready for local VS0 acceptance." --state-dir data/quickstart-vs0 --json
cornerstone action dry-run <action_id> --state-dir data/quickstart-vs0 --json
cornerstone audit verify --state-dir data/quickstart-vs0 --json
cornerstone scenario verify vs0-runtime-acceptance --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json --json
```

Placeholder IDs are for manual walkthrough; scenario PASS requires generated transcripts. The executable quickstart proves structural behavior only — not product value, production tenancy, live providers, or human usability.

## Verification Reference

- Documentation wiring: `scripts/verify_sot_docs.sh` (chains CLI-native-first, local-verification-plane, design-system, and scaffold-readiness doc checks).
- Scenario gates: `cornerstone scenario verify <set> --json` (e.g. `vs0-universal-core`, `vs4-product-alpha-ui-daily-loop`), `make verify-local-fast`.
- Release rule: no PASS without concrete scenario evidence; no product-value claim without Plane 2 evidence; the verdict never exceeds the weakest required row.
- CLI-native-first rule remains: no feature scenario passes without its native `cornerstone ...` path.

## Documentation Map

Authority order and the full index live in `docs/sot/README.md`. Key entries:

| Document | Role |
|---|---|
| `docs/adr/ADR-0007-product-value-first-reset.md` | The 2026-07-04 direction decision |
| `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` | Product identity: active spine (Part 0) + labeled long-term direction |
| `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` | Long-term behavior standard (206 scenarios) + family activity status |
| `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` | Plane 2: product-value acceptance (CS-VAL) and verdict ladder |
| `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md` | Active milestone |
| `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md` | Plane 1 local verification machinery |
| `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md` | Design system (doctrine: Calm Surface. Deep Evidence. Safe Action.) |
| `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` | Historical: VS-0 setup-planning gate |
| `docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md` | Historical: scaffold-gate context |

Historical VS0–VS4 contracts, slices, and verification reports remain under `docs/scenario-contracts/` and `docs/verification-reports/` as the structural evidence record. They are history, not the product story.

## Design Doctrine

**Calm Surface. Deep Evidence. Safe Action.** Light-first calm workspace; evidence, policy, and audit as progressively disclosed detail; no chatbot-only, admin-first, or dark command-center default. Internal tokens (record IDs, policy strings, trust-state enums) must not appear as user-facing copy — user-facing language uses Drop, Brief, Source, Decision, Review, Inbox, History.
