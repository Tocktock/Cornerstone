# Chain of Custody Governance Decision Record

Document id: hpcc-v1-chain-of-custody-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Chain of Custody means the ordered record of accountable handoffs from pack-out through patient site receipt.
The governance group reviewed the effect of Chain of Custody on GDP Audit Readiness.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The compliance forum decided that courier app signatures are acceptable only when paired with device identity and route geofence evidence.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Compliance.

## Rationale
Custody completeness is measured by signed handoffs, timestamp consistency, geofence match, and identity verification.
The policy requires custody evidence to be immutable after release except through a correction record.
Every custody handoff must include actor identity, location, timestamp, package seal state, and exception notes when applicable.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a depot supervisor handoff without seal state is incomplete even when the driver signature is present.
The main risk is relying on carrier milestone labels that do not identify the accountable actor.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should custody gaps automatically reduce the graph trust label for affected concepts?
Should Chain of Custody be visible in customer-facing answers or restricted to internal quality review?
