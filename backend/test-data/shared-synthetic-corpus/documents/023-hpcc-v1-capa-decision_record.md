# CAPA Governance Decision Record

Document id: hpcc-v1-capa-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
CAPA means a corrective and preventive action record that addresses root cause, containment, verification, and recurrence prevention.
The governance group reviewed the effect of CAPA on Deviation Triage.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The CAPA council decided that training-only actions are insufficient for recurring lane qualification failures.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Quality Systems.

## Rationale
CAPA effectiveness is measured by repeat-event rate, verification evidence quality, owner timeliness, and control durability.
The policy requires preventive actions to name the control that will detect recurrence before patient impact.
A CAPA must link to the triggering deviation, the root cause statement, effectiveness criteria, and closure evidence.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a revised carrier checklist is weak unless paired with audit sampling and route risk score recalibration.
The main risk is writing corrective actions that address symptoms but not route or packaging control design.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should CAPA effectiveness be visible in the official ontology graph?
Should CAPA be visible in customer-facing answers or restricted to internal quality review?
