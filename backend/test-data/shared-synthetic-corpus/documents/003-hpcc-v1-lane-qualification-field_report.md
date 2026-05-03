# Lane Qualification Field Evidence Report

Document id: hpcc-v1-lane-qualification-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Lane Qualification.
Lane Qualification means a documented approval that a transport lane can protect product quality under expected seasonal, carrier, customs, and handoff conditions.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, the Seoul to Singapore oncology lane uses active monitoring during airport dwell because the customs hold risk is material.
The lane qualification score is the weighted sum of seasonal thermal margin, carrier handoff reliability, customs dwell exposure, and recovery access.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
A lane must not be released for commercial biologic movement until a qualification packet includes route risk scoring, validated shipper fit, and escalation contacts.
The policy requires every qualified lane to be rechecked before summer and winter operating windows.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is silent route drift after a carrier subcontractor changes the airport handoff pattern.
For instance, the Seoul to Singapore oncology lane uses active monitoring during airport dwell because the customs hold risk is material.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should lane qualification include a separate score for weekend customs staffing?
Should this field report remain evidence-only after a decision record cites it?
