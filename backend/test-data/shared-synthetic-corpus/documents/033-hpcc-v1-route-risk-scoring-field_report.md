# Route Risk Scoring Field Evidence Report

Document id: hpcc-v1-route-risk-scoring-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Route Risk Scoring.
Route Risk Scoring means the calculation that estimates product quality risk across route, weather, customs, carrier, and contingency factors.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a short route can have high risk when airport dwell occurs during a heat advisory.
The score uses thermal margin, dwell exposure, carrier reliability, contingency depot access, and seasonal volatility.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
A high route risk score must trigger either enhanced monitoring, a stronger shipper, or a quality-approved exception.
The policy requires score inputs to be auditable and reproducible from source evidence.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is allowing planners to override risk factors without a decision record.
For instance, a short route can have high risk when airport dwell occurs during a heat advisory.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should route risk scoring be recalculated after every deviation or only at lane review?
Should this field report remain evidence-only after a decision record cites it?
