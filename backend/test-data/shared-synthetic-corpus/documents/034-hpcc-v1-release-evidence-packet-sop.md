# Release Evidence Packet Control SOP

Document id: hpcc-v1-release-evidence-packet-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Release Evidence Packet means the collected evidence used to justify shipment disposition and downstream release state.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Release Evidence Packet is Quality Assurance.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires release packets to preserve source evidence links instead of copied summary text only.
A release evidence packet must include stability budget calculation, custody evidence, sensor files, deviation records, and final disposition.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Packet readiness is measured by required evidence completeness, reviewer signoff, exception closure, and unresolved question count.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a packet can cite a CAPA for prevention but still needs shipment-specific stability evidence.
The main risk is approving release from a narrative summary without the raw evidence trail.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should the release evidence packet become the default graph focus for audit queries?
Can the reviewer trace every official claim about Release Evidence Packet to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
