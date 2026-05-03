# Product Positioning: Cornerstone vs Chatbot, RAG, Search, Wiki, and Knowledge Graph

Cornerstone can work with chatbots, RAG systems, search, and wikis. It is not the same product category as any one of them.

## Comparison table

| System | What it does well | Main weakness | Cornerstone difference |
|---|---|---|---|
| Wiki | Stores long-form organizational pages. | Meaning is scattered across pages and often stale. | Converts evidence into reviewed Concepts and Relations. |
| Search | Finds documents quickly. | Does not decide which claim is official. | Serves official graph truth with citations. |
| RAG chatbot | Produces natural-language answers from retrieved context. | Can blend claims or imply unsupported certainty. | Provides reviewed graph context and trust labels for safer answers. |
| Vector database | Retrieves semantically similar chunks. | Similarity is not truth. | Uses evidence, review, and officialization gates. |
| Traditional knowledge graph | Represents structured relationships. | Often expensive to maintain manually. | Uses source connectors and candidate extraction to accelerate review. |
| Cornerstone | Builds reviewed organizational meaning. | Requires review workflow and evidence discipline. | Makes the official ontology graph explainable and auditable. |

## The key distinction

Search asks:

```text
Which documents mention settlement?
```

A chatbot asks:

```text
What answer can be generated from retrieved text?
```

Cornerstone asks:

```text
What is the reviewed official Concept, how is it related to other Concepts, and what evidence proves it?
```

## How Cornerstone complements AI assistants

AI assistants need high-quality context.

Instead of asking an assistant to infer meaning from scattered documents, a developer can give the assistant official graph context:

```text
Concept: Settlement
Definition: reviewed definition
Relations: direct official graph edges
Citations: reviewed EvidenceFragments
Trust label: official
Limitations: pending candidates excluded
```

That makes downstream AI behavior safer and easier to audit.

## Why review is not optional

Without review, the system becomes an automatic graph generator.

That would weaken the product because users could not distinguish:

```text
- something the source says
- something the extractor guessed
- something a reviewer approved
- something the organization treats as official
```

Cornerstone's product value depends on preserving those distinctions.

## Positioning acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-POS-01 | Compares Cornerstone with common alternatives. | Table covers wiki, search, RAG chatbot, vector DB, knowledge graph. | complete |
| PROD-POS-02 | Key distinction is simple. | Contrasts search, chatbot, and Cornerstone questions. | complete |
| PROD-POS-03 | AI-assistant value is explained. | Shows official graph context for assistants. | complete |
| PROD-POS-04 | Review importance is explicit. | States automatic graph generation would weaken trust. | complete |
| PROD-POS-05 | Product category is clear. | Names reviewed organizational meaning as the main value. | complete |
