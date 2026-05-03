# Chain of Custody Control SOP

Document id: hpcc-v1-chain-of-custody-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Chain of Custody means the ordered record of accountable handoffs from pack-out through patient site receipt.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Chain of Custody is Compliance.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires custody evidence to be immutable after release except through a correction record.
Every custody handoff must include actor identity, location, timestamp, package seal state, and exception notes when applicable.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Custody completeness is measured by signed handoffs, timestamp consistency, geofence match, and identity verification.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a depot supervisor handoff without seal state is incomplete even when the driver signature is present.
The main risk is relying on carrier milestone labels that do not identify the accountable actor.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should custody gaps automatically reduce the graph trust label for affected concepts?
Can the reviewer trace every official claim about Chain of Custody to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
