# Calibration Drift Control SOP

Document id: hpcc-v1-calibration-drift-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Calibration Drift means movement of a measurement device away from accepted tolerance between calibration events.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Calibration Drift is Metrology.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires paired sensor comparison when drift affects a shipment with limited stability budget.
A sensor with unresolved calibration drift must not be used as the sole evidence for product release.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Drift risk is measured by days since calibration, observed variance, device family history, and excursion decision dependency.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a logger reading 7.8 C may not prove compliance if its paired sensor reads 8.4 C and drift is unresolved.
The main risk is using a single sensor trace without checking calibration certificate status.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should calibration drift automatically downgrade evidence freshness or trust state?
Can the reviewer trace every official claim about Calibration Drift to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
