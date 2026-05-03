# Quarantine Workflow Governance Decision Record

Document id: hpcc-v1-quarantine-workflow-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Quarantine Workflow means the physical and system segregation process for units that cannot be released until quality disposition is complete.
The governance group reviewed the effect of Quarantine Workflow on Quality Hold.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The warehouse quality team decided that scanner override is prohibited for quarantined biologic inventory.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Warehouse Quality.

## Rationale
Quarantine performance is measured by segregation accuracy, hold aging, scan compliance, and release reconciliation.
The policy requires physical labels and system status to agree before a unit is considered quarantined.
Quarantine workflow must prevent pick, pack, transfer, billing, and patient dispatch for affected units.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a unit can be physically in the quarantine cage but still unsafe if warehouse management status is available.
The main risk is a manual inventory move that bypasses the quality hold state.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should quarantine workflow publish events to patient scheduling when a critical dose is affected?
Should Quarantine Workflow be visible in customer-facing answers or restricted to internal quality review?
