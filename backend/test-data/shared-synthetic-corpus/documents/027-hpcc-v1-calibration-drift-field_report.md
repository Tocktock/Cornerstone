# Calibration Drift Field Evidence Report

Document id: hpcc-v1-calibration-drift-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Calibration Drift.
Calibration Drift means movement of a measurement device away from accepted tolerance between calibration events.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a logger reading 7.8 C may not prove compliance if its paired sensor reads 8.4 C and drift is unresolved.
Drift risk is measured by days since calibration, observed variance, device family history, and excursion decision dependency.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
A sensor with unresolved calibration drift must not be used as the sole evidence for product release.
The policy requires paired sensor comparison when drift affects a shipment with limited stability budget.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is using a single sensor trace without checking calibration certificate status.
For instance, a logger reading 7.8 C may not prove compliance if its paired sensor reads 8.4 C and drift is unresolved.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should calibration drift automatically downgrade evidence freshness or trust state?
Should this field report remain evidence-only after a decision record cites it?
