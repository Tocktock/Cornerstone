# Deviation Triage Governance Decision Record

Document id: hpcc-v1-deviation-triage-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Deviation Triage means the first structured assessment that classifies a quality event, assigns ownership, and selects the disposition path.
The governance group reviewed the effect of Deviation Triage on CAPA.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The quality operations team decided that triage can close low-risk paperwork deviations only when no product exposure uncertainty exists.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Quality Operations.

## Rationale
Triage quality is measured by classification accuracy, evidence completeness, owner assignment, and time to first disposition.
The policy requires triage notes to separate observed facts from inferred hypotheses.
Deviation triage must identify whether the event affects product quality, patient scheduling, regulatory reporting, or only operational performance.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a missing receiver signature is paperwork-only when custody is otherwise proven by geofence and device evidence.
The main risk is closing a deviation before stability budget and custody evidence are reviewed together.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should triage confidence be computed by rule or assigned by the reviewer?
Should Deviation Triage be visible in customer-facing answers or restricted to internal quality review?
