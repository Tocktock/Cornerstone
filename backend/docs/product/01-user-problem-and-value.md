# User Problem and Product Value

## The user problem

Teams have knowledge, but the meaning of that knowledge is scattered.

A settlement operations team may have:

```text
- a policy document defining settlement
- a finance runbook explaining reconciliation
- a product note describing ledger updates
- a risk document mentioning exceptions
- a manager who knows which source is official
```

When someone asks “What is settlement?”, the answer is not only a definition. The user also needs to know:

```text
- which definition is official
- what processes settlement depends on
- what settlement changes
- who reviewed the answer
- what source proves the answer
- whether there are pending or conflicting updates
```

## Why ordinary documents are not enough

Documents are useful, but they are not structured truth.

They can be:

```text
- duplicated
- outdated
- written with inconsistent terms
- ambiguous about ownership
- hard to connect to related concepts
- difficult for AI systems to use safely
```

A document can say something true, but the organization still needs a reviewed structure that says:

```text
This is the official Concept.
This is how it relates to other Concepts.
This evidence supports the claim.
This is safe to serve.
```

## Why ordinary chatbots are not enough

A chatbot can summarize documents, but summarization alone does not create governance.

A chatbot may answer:

```text
Settlement is probably the process of finalizing payments.
```

Cornerstone should answer:

```text
Settlement is the reviewed official Concept for finalizing obligations.
It is supported by these reviewed EvidenceFragments.
Its direct Relations are Clearing precedes Settlement and Settlement updates Ledger.
Pending candidates are not included in this official graph.
```

The product difference is reviewable truth, not fluent text.

## Product value

Cornerstone creates value in five ways.

### 1. Explainability

Every official graph claim can point back to evidence.

```text
Concept: Settlement
Evidence: source fragment from Settlement Operations Guide
Reviewer: reviewer@example.com
Status: official
```

### 2. Better Single Source of Truth

The official truth is not hidden in one large document. It is represented as Concepts and Relations.

```text
Settlement --updates--> Ledger
Clearing --precedes--> Settlement
Reconciliation --validates--> Settlement
```

### 3. Safer AI context

AI systems can use the official graph instead of guessing from raw text.

### 4. Human control

The system can propose concepts and relations, but reviewers decide what becomes official.

### 5. Operational readiness

Operators can run proof and readiness checklists to confirm whether a focus concept is safe to serve.

## Product value acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-VALUE-01 | User pain is concrete. | Uses settlement operations example. | complete |
| PROD-VALUE-02 | Document limitations are clear. | Explains ambiguity, duplication, and lack of structure. | complete |
| PROD-VALUE-03 | Chatbot limitation is clear. | Distinguishes summarization from reviewed truth. | complete |
| PROD-VALUE-04 | Product value is measurable. | Lists explainability, SSOT, safer AI, human control, readiness. | complete |
| PROD-VALUE-05 | Product value maps to ontology objects. | Uses Concepts, Relations, EvidenceFragments, and official status. | complete |
