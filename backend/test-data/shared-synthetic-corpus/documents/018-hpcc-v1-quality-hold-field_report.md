# Quality Hold Field Evidence Report

Document id: hpcc-v1-quality-hold-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Quality Hold.
Quality Hold means a controlled state that prevents product release until quality evidence is reviewed and disposition is recorded.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a custody gap and a sensor gap can share one quality hold when they affect the same physical shipment.
Hold aging is measured by elapsed hours, missing evidence count, patient impact tier, and next required reviewer.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
A shipment under quality hold must not be released to inventory, patient scheduling, or billing systems.
The policy requires quality hold reason codes to reference the triggering evidence and the required disposition path.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is a downstream system clearing the hold because carrier delivery is complete.
For instance, a custody gap and a sensor gap can share one quality hold when they affect the same physical shipment.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should quality hold state expire if no reviewer acts within the service-level target?
Should this field report remain evidence-only after a decision record cites it?
