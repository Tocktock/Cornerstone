# Quarantine Workflow Control SOP

Document id: hpcc-v1-quarantine-workflow-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Quarantine Workflow means the physical and system segregation process for units that cannot be released until quality disposition is complete.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Quarantine Workflow is Warehouse Quality.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires physical labels and system status to agree before a unit is considered quarantined.
Quarantine workflow must prevent pick, pack, transfer, billing, and patient dispatch for affected units.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Quarantine performance is measured by segregation accuracy, hold aging, scan compliance, and release reconciliation.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a unit can be physically in the quarantine cage but still unsafe if warehouse management status is available.
The main risk is a manual inventory move that bypasses the quality hold state.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should quarantine workflow publish events to patient scheduling when a critical dose is affected?
Can the reviewer trace every official claim about Quarantine Workflow to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
