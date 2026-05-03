# CAPA Field Evidence Report

Document id: hpcc-v1-capa-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to CAPA.
CAPA means a corrective and preventive action record that addresses root cause, containment, verification, and recurrence prevention.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a revised carrier checklist is weak unless paired with audit sampling and route risk score recalibration.
CAPA effectiveness is measured by repeat-event rate, verification evidence quality, owner timeliness, and control durability.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
A CAPA must link to the triggering deviation, the root cause statement, effectiveness criteria, and closure evidence.
The policy requires preventive actions to name the control that will detect recurrence before patient impact.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is writing corrective actions that address symptoms but not route or packaging control design.
For instance, a revised carrier checklist is weak unless paired with audit sampling and route risk score recalibration.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should CAPA effectiveness be visible in the official ontology graph?
Should this field report remain evidence-only after a decision record cites it?
