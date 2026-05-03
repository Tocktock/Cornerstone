# FEFO Allocation Field Evidence Report

Document id: hpcc-v1-fefo-allocation-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to FEFO Allocation.
FEFO Allocation means selecting inventory by earliest usable expiry while respecting quality state, route feasibility, and patient scheduling constraints.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, the earliest expiring lot cannot be allocated if its release evidence packet is incomplete.
FEFO effectiveness is measured by expiry waste, release eligibility, patient promise adherence, and avoidable exception rate.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
FEFO allocation must exclude units on quality hold and units without enough stability budget for the planned lane.
The policy requires allocation logic to treat release eligibility as stronger than expiry priority.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is optimizing expiry while ignoring route risk and quality hold state.
For instance, the earliest expiring lot cannot be allocated if its release evidence packet is incomplete.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should FEFO allocation consume graph answers directly or only read materialized release state?
Should this field report remain evidence-only after a decision record cites it?
