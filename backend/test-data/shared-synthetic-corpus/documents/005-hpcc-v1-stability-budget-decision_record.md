# Stability Budget Governance Decision Record

Document id: hpcc-v1-stability-budget-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Stability Budget means the remaining approved exposure allowance for a product lot after considering temperature, duration, and product-specific stability evidence.
The governance group reviewed the effect of Stability Budget on FEFO Allocation.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The quality council decided that estimated budget recovery is prohibited unless the stability memo explicitly names the recovery method.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Product Quality.

## Rationale
The budget is tracked in minutes by temperature band and is reduced only by reviewed excursion evidence.
The policy requires budget calculation to use the product stability memo, not the shipment average temperature.
A release decision must preserve a nonnegative stability budget for every shipped unit and every documented handoff.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a ten minute 8.7 C event may consume more budget than a thirty minute 7.9 C event for a narrow 2-8 C biologic.
The main risk is aggregating sensor readings before applying the product-specific temperature band.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should the stability budget service expose remaining budget as a graph edge attribute?
Should Stability Budget be visible in customer-facing answers or restricted to internal quality review?
