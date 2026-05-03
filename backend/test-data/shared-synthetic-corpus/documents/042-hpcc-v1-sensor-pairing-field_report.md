# Sensor Pairing Field Evidence Report

Document id: hpcc-v1-sensor-pairing-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Sensor Pairing.
Sensor Pairing means assigning two compatible measurement devices to a shipment so readings can validate each other and expose drift or placement error.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a top-cavity logger and payload-core logger can reveal a local placement artifact.
Pairing reliability is measured by device compatibility, placement separation, time synchronization, and drift agreement.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
Sensor pairing must be used when shipment value, stability budget, or regulatory category requires redundant evidence.
The policy requires paired sensors to be time-synchronized before pack-out and reconciled after receipt.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is pairing two devices from the same suspect calibration batch.
For instance, a top-cavity logger and payload-core logger can reveal a local placement artifact.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should paired sensor disagreement create automatic deviation triage?
Should this field report remain evidence-only after a decision record cites it?
