# Deviation Triage Field Evidence Report

Document id: hpcc-v1-deviation-triage-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Deviation Triage.
Deviation Triage means the first structured assessment that classifies a quality event, assigns ownership, and selects the disposition path.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a missing receiver signature is paperwork-only when custody is otherwise proven by geofence and device evidence.
Triage quality is measured by classification accuracy, evidence completeness, owner assignment, and time to first disposition.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
Deviation triage must identify whether the event affects product quality, patient scheduling, regulatory reporting, or only operational performance.
The policy requires triage notes to separate observed facts from inferred hypotheses.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is closing a deviation before stability budget and custody evidence are reviewed together.
For instance, a missing receiver signature is paperwork-only when custody is otherwise proven by geofence and device evidence.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should triage confidence be computed by rule or assigned by the reviewer?
Should this field report remain evidence-only after a decision record cites it?
