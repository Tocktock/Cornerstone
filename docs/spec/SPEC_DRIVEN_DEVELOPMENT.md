# Spec-driven Development

**Status:** Approved  
**Type:** Delivery Operating Model  
**Last Updated:** 2026-04-05

---

## 1. Core principle

Cornerstone’s unit of delivery is **not code**.  
Cornerstone’s unit of delivery is a **specification-backed capability**.

The expected order is:

> problem definition → spec → design → decision record → implementation → verification → documentation update

Code-first delivery is allowed only for trivial local changes that do not alter capability, domain behavior, invariants, or interfaces.

---

## 2. Why this operating model exists

Cornerstone deals with:
- meaning
- provenance
- governance
- reviewability
- durable organizational context

These are easy to damage with ad hoc implementation-first work.

Spec-driven development exists to make irreversible or high-impact decisions visible **before** implementation hardens them.

---

## 3. Document types

### 3.1 Product SoT
Defines identity, boundaries, purpose, and non-negotiable product principles.  
Example: `sot/PROJECT_SOT.md`

### 3.2 Why / Goals
Defines the real-world problem, intended outcome, and North Star.  
Example: `sot/WHY_AND_GOALS.md`

### 3.3 Product Spec
Defines v1/v2 capability scope and product requirements.  
Example: `spec/PRODUCT_SPEC.md`

### 3.4 Domain Spec
Defines entities, relationships, lifecycle rules, and invariants.  
Example: `spec/DOMAIN_MODEL.md`

### 3.5 Feature Spec
Defines one concrete capability or capability slice for delivery.

### 3.6 Technical Design
Defines implementation structure, data flow, operational design, and technical trade-offs.

### 3.7 Decision Record
Captures durable rationale for a choice that materially affects boundaries, domain behavior, or architecture.

### 3.8 Runbook
Captures operational procedures such as sync recovery, incident handling, or manual review steps.

---

## 4. Authority hierarchy

If documents conflict, the higher layer wins.

1. Product SoT
2. Why / Goals
3. Product Spec
4. Domain Spec
5. Replaceable vs Non-replaceable
6. Delivery Operating Model
7. Glossary
8. Feature Spec
9. Technical Design
10. Implementation

---

## 5. Unit of work

A meaningful delivery unit should include:

- one **Feature Spec**
- one **Technical Design** when needed
- one or more **Decision Records** when boundaries or durable trade-offs are involved
- one explicit **Acceptance / Verification plan**

A task ticket is not enough by itself.  
The minimum acceptable unit is a **verifiable spec bundle**.

---

## 6. Required sections for every feature spec

Every feature spec must include the following.

1. Summary
2. Background / Problem
3. Goals
4. Non-goals
5. Users / Consumers
6. Scope
7. Domain Impact
8. Requirements
9. Non-replaceable Invariants
10. Replaceable Choices
11. Acceptance Criteria
12. Risks
13. Open Questions
14. Rollout / Verification

---

## 7. State machine for specs

Recommended lifecycle:

1. **Draft**  
   Problem and direction are still being shaped.

2. **Review**  
   Product, domain, and technical review are in progress.

3. **Approved**  
   Delivery may begin.

4. **In Delivery**  
   Implementation is underway.

5. **Verified**  
   Acceptance criteria are met and checked.

6. **Adopted**  
   The capability is operating in the intended environment.

7. **Deprecated**  
   The spec no longer represents the active capability.

---

## 8. Review gates

### 8.1 Before approval
Before a spec reaches `Approved`, the following should be checked:

- alignment with `sot/PROJECT_SOT.md`
- alignment with `spec/DOMAIN_MODEL.md`
- whether any non-replaceable concept is touched
- whether a `DecisionRecord` is required
- whether acceptance criteria are observable and testable

### 8.2 Before verification
Before a spec reaches `Verified`, confirm:

- acceptance criteria were actually exercised
- operational failure paths were considered
- relevant documentation was updated
- rollout and fallback behavior were reviewed

---

## 9. Rules for irreversible change

Do not finalize irreversible schema, protocol, or operating decisions before the relevant spec is approved.

Examples of irreversible or high-cost changes:
- entity model changes
- status semantics changes
- reviewability model changes
- transport contracts adopted by external consumers
- source-of-truth semantics
- storage layout that hardcodes the wrong abstraction

If these are touched, the work requires explicit specification.

---

## 10. Definition of done

A feature is not done when code merges.  
A feature is done when:

- the approved spec exists
- the implementation matches the approved spec
- the acceptance criteria are verified
- docs are updated
- any required decision record is linked
- the capability is safe to operate or clearly staged

---

## 11. Fast-path rule

Small implementation-only changes may skip a new feature spec if all of the following are true:

- no domain entity changes
- no interface changes
- no change to reviewability, provenance, or officialization
- no change to non-replaceable concepts
- no new operational risk class

If any of these are false, write a spec first.

---

## 12. Operating reminder

Cornerstone is building durable organizational context.  
Spec-driven delivery is part of the product strategy, not just process overhead.
