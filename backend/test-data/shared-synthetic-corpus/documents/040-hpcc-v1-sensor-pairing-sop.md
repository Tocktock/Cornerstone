# Sensor Pairing Control SOP

Document id: hpcc-v1-sensor-pairing-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Sensor Pairing means assigning two compatible measurement devices to a shipment so readings can validate each other and expose drift or placement error.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Sensor Pairing is Metrology.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires paired sensors to be time-synchronized before pack-out and reconciled after receipt.
Sensor pairing must be used when shipment value, stability budget, or regulatory category requires redundant evidence.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Pairing reliability is measured by device compatibility, placement separation, time synchronization, and drift agreement.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a top-cavity logger and payload-core logger can reveal a local placement artifact.
The main risk is pairing two devices from the same suspect calibration batch.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should paired sensor disagreement create automatic deviation triage?
Can the reviewer trace every official claim about Sensor Pairing to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
