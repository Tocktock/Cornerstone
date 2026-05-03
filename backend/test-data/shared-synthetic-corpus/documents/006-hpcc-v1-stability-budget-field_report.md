# Stability Budget Field Evidence Report

Document id: hpcc-v1-stability-budget-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Stability Budget.
Stability Budget means the remaining approved exposure allowance for a product lot after considering temperature, duration, and product-specific stability evidence.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a ten minute 8.7 C event may consume more budget than a thirty minute 7.9 C event for a narrow 2-8 C biologic.
The budget is tracked in minutes by temperature band and is reduced only by reviewed excursion evidence.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
A release decision must preserve a nonnegative stability budget for every shipped unit and every documented handoff.
The policy requires budget calculation to use the product stability memo, not the shipment average temperature.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is aggregating sensor readings before applying the product-specific temperature band.
For instance, a ten minute 8.7 C event may consume more budget than a thirty minute 7.9 C event for a narrow 2-8 C biologic.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should the stability budget service expose remaining budget as a graph edge attribute?
Should this field report remain evidence-only after a decision record cites it?
