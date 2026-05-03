# Quarantine Workflow Field Evidence Report

Document id: hpcc-v1-quarantine-workflow-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Quarantine Workflow.
Quarantine Workflow means the physical and system segregation process for units that cannot be released until quality disposition is complete.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a unit can be physically in the quarantine cage but still unsafe if warehouse management status is available.
Quarantine performance is measured by segregation accuracy, hold aging, scan compliance, and release reconciliation.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
Quarantine workflow must prevent pick, pack, transfer, billing, and patient dispatch for affected units.
The policy requires physical labels and system status to agree before a unit is considered quarantined.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is a manual inventory move that bypasses the quality hold state.
For instance, a unit can be physically in the quarantine cage but still unsafe if warehouse management status is available.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should quarantine workflow publish events to patient scheduling when a critical dose is affected?
Should this field report remain evidence-only after a decision record cites it?
