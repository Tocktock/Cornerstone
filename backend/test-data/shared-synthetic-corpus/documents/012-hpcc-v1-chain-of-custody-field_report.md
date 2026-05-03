# Chain of Custody Field Evidence Report

Document id: hpcc-v1-chain-of-custody-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Chain of Custody.
Chain of Custody means the ordered record of accountable handoffs from pack-out through patient site receipt.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a depot supervisor handoff without seal state is incomplete even when the driver signature is present.
Custody completeness is measured by signed handoffs, timestamp consistency, geofence match, and identity verification.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
Every custody handoff must include actor identity, location, timestamp, package seal state, and exception notes when applicable.
The policy requires custody evidence to be immutable after release except through a correction record.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is relying on carrier milestone labels that do not identify the accountable actor.
For instance, a depot supervisor handoff without seal state is incomplete even when the driver signature is present.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should custody gaps automatically reduce the graph trust label for affected concepts?
Should this field report remain evidence-only after a decision record cites it?
