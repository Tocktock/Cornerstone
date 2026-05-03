# FEFO Allocation Governance Decision Record

Document id: hpcc-v1-fefo-allocation-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
FEFO Allocation means selecting inventory by earliest usable expiry while respecting quality state, route feasibility, and patient scheduling constraints.
The governance group reviewed the effect of FEFO Allocation on Stability Budget.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The inventory council decided that patient-critical shipments may bypass normal FEFO order only with a decision record.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Inventory Operations.

## Rationale
FEFO effectiveness is measured by expiry waste, release eligibility, patient promise adherence, and avoidable exception rate.
The policy requires allocation logic to treat release eligibility as stronger than expiry priority.
FEFO allocation must exclude units on quality hold and units without enough stability budget for the planned lane.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, the earliest expiring lot cannot be allocated if its release evidence packet is incomplete.
The main risk is optimizing expiry while ignoring route risk and quality hold state.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should FEFO allocation consume graph answers directly or only read materialized release state?
Should FEFO Allocation be visible in customer-facing answers or restricted to internal quality review?
