# Lane Qualification Governance Decision Record

Document id: hpcc-v1-lane-qualification-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Lane Qualification means a documented approval that a transport lane can protect product quality under expected seasonal, carrier, customs, and handoff conditions.
The governance group reviewed the effect of Lane Qualification on CAPA.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The review board decided that any lane with two unresolved excursion near misses in ninety days moves back to pilot status.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Quality Operations.

## Rationale
The lane qualification score is the weighted sum of seasonal thermal margin, carrier handoff reliability, customs dwell exposure, and recovery access.
The policy requires every qualified lane to be rechecked before summer and winter operating windows.
A lane must not be released for commercial biologic movement until a qualification packet includes route risk scoring, validated shipper fit, and escalation contacts.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, the Seoul to Singapore oncology lane uses active monitoring during airport dwell because the customs hold risk is material.
The main risk is silent route drift after a carrier subcontractor changes the airport handoff pattern.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should lane qualification include a separate score for weekend customs staffing?
Should Lane Qualification be visible in customer-facing answers or restricted to internal quality review?
