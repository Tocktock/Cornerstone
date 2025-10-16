# Improving the Keyword Explorer Pipeline

## Current Implementation and Its Limitations

Following the first wave of improvements, the Keyword Explorer now performs the Stage 1 chunk preparation and Stage 2 concept extraction described in this document. Incoming payloads are normalised into `KeywordSourceChunk` objects, and each chunk is analysed with a hybrid pipeline (n‑gram scoring, RAKE-style statistics, embedding similarity, and optional LLM summaries) that produces rich `ConceptCandidate` entries complete with coverage metrics and score breakdowns. Stage 2 has extensive instrumentation (`keyword.stage2.*` logs, debug payloads, and per-component timing) and is fully configurable through the new `KEYWORD_STAGE2_*` environment variables.

With Stage 3 clustering, optional LLM labelling, Stage 4 re-ranking, Stage 5 harmonisation, and Stage 6 verification all wired into the API, the Explorer now returns harmonised, LLM-reviewed semantic concepts by default. Remaining work focuses on polish (caching Stage 2 embeddings, expanding automated tests, guarding malformed LLM outputs) and on defining the next evolution of the pipeline.

## Goals for an Improved Pipeline

To address these issues, we propose a new keyword exploration pipeline that is more robust and semantically aware. The goals of the improved design are:

- **Capture core domain concepts:** Extract meaningful multi-word phrases and high-level themes that truly represent the domain and business context, rather than just frequent tokens. This includes surfacing product names, feature names, process names, issue categories, and other key concepts (possibly in both English and Korean, as the system is bilingual).
- **Move beyond raw frequency:** Avoid ranking terms solely by occurrence counts. Incorporate semantic significance—for example, consider how unique or representative a term is for the corpus. Rare but important terms should be picked up, and common but trivial terms should be down-weighted.
- **Leverage the existing stack:** Design the pipeline to make smart use of the current LLM and embedding tools, rather than introducing entirely new model dependencies. Focus on orchestrating the agent flow and combining algorithms, instead of simply swapping in a bigger model.
- **Prioritize architecture and maintainability:** Break the extraction process into clear stages (e.g., text preprocessing, candidate generation, concept consolidation, re-ranking) that are easy to understand and maintain. Each stage should have a clear purpose and be testable. This modularity will make the system more resilient and easier to extend.
- **Ensure resilience and fallbacks:** Provide sensible defaults and fallbacks. If the LLM is unavailable or a certain algorithm does not produce output, the system should still return a reasonable set of keywords rather than failing silently. Logging and debug information (similar to the current `debug_payload`) should be retained to trace how keywords were picked.

With these goals in mind, we outline an improved pipeline below.

## Proposed Pipeline Design

Instead of a single pass of “count words, then maybe filter by LLM,” the new pipeline will have multiple stages, each enhancing the quality of the keywords:

1. Document chunking and preparation
2. Candidate concept extraction (using hybrid methods such as embeddings and statistical phrases)
3. Concept consolidation and clustering
4. Re-ranking and core concept selection
5. Canonical label harmonisation (post-process)
6. LLM-based refinement (optional)
7. Insight summarisation and reporting (optional)

Each stage is described in detail in the following subsections.

### Stage 1: Document Chunking & Preparation

✅ **Status:** implemented.

Cornerstone now normalises ingestion payloads into `KeywordSourceChunk` objects during Stage 1. Each chunk keeps the raw text, a language-aware normalised version, metadata (document IDs, section paths, headings), and token counts for later heuristics. The helper functions added to `keywords.py` collapse whitespace, strip punctuation for English, and preserve Hangul phrases so multi-word candidates survive into Stage 2. The stage also tracks per-project diagnostics (total payloads processed, skipped empty entries, language mix) that surface in the debug payload.

Future refinements (post Stage 3) could introduce hierarchical chunking or richer metadata, but the base preparation work is in place.

### Stage 2: Candidate Concept Extraction

✅ **Status:** implemented.

The extraction function (`extract_concept_candidates`) now orchestrates the hybrid approach envisioned here:

- **Embedding-based keyphrase scoring:** For each chunk the existing `EmbeddingService` computes a chunk vector (OpenAI or Ollama) and reuses it to rank top n‑gram phrases by cosine similarity. Phrase vectors are cached so later stages can reuse them.
- **Statistical phrase scoring:** We run RAKE-like scoring over the token stream to surface domain-specific collocations and blend the results into the candidate aggregate. We can adjust how many statistical phrases to keep per chunk.
- **LLM summaries:** When enabled, the `KeywordLLMFilter` can request short concept summaries from representative chunks, seeding additional generated candidates. This can be toggled per deployment.

All of these knobs are exposed via environment variables—for example `KEYWORD_STAGE2_MAX_NGRAM`, `KEYWORD_STAGE2_MAX_EMBEDDING_PHRASES`, `KEYWORD_STAGE2_USE_LLM_SUMMARY`, and the weighting trio (`KEYWORD_STAGE2_EMBEDDING_WEIGHT`, `…_STATISTICAL_WEIGHT`, `…_LLM_WEIGHT`). Stage 2 captures per-component timings, backend identifiers, and top candidates in the debug payload, making it easier to diagnose which backend (OpenAI vs. Ollama) was used.

To keep rare-but-important concepts from being drowned out, the defaults now lean semantic: `KEYWORD_STAGE2_EMBEDDING_WEIGHT=2.25` and `KEYWORD_STAGE2_LLM_WEIGHT=3.1`. Lower those values if you prefer frequency-first scoring.

Candidates now retain their averaged embedding vectors (plus backend identifiers), letting Stage 3 reuse the cached embeddings rather than re-querying the LLM backend on the next hop.

Remaining Stage 2 follow-ups: none—defensive tests now cover malformed LLM responses and embedding reuse is cached.

### Stage 3: Concept Consolidation and Clustering

✅ **Status:** implemented (embedding-aware clustering with optional LLM labeling).

After gathering candidates, we often end up with semantically overlapping terms (e.g., “login issue,” “sign-in problem”) or different granularities (specific error codes vs. broader categories). This stage groups and condenses the candidate list into core concepts:

- **Clustering by semantic similarity:** Compute embeddings for each candidate phrase (we may reuse those from the KeyBERT step) and cluster them (e.g., with K-means, HDBSCAN, or a simple similarity threshold). This groups near-duplicates or synonyms, ensuring we do not present multiple variants of the same concept.
- **Topic modeling:** Run a topic modeling approach (such as LDA, NMF, or BERTopic) across all documents to discover themes. Topic representatives like “Payment Issues” or “Account Management” provide high-level context. This step can be run offline to guide clustering.
- **Consolidation into representative concepts:** For each cluster of similar candidates, choose a representative term or phrase. In the current implementation we pick the highest-scoring member by default and, when `KEYWORD_STAGE3_LABEL_CLUSTERS=true`, ask the configured chat backend to suggest a concise cluster label and optional description via `KeywordLLMFilter.label_clusters`. Results surface aliases, label provenance, and debug payloads for traceability. Disable or cap labeling with `KEYWORD_STAGE3_LABEL_MAX_CLUSTERS`.

Stage 3 now reuses the embedding vectors attached to each candidate, so when Ollama (or OpenAI) already processed the phrase during Stage 2 we skip the redundant embed call. Only concepts missing vectors trigger additional requests, cutting per-search latency.

At the end of Stage 3, we have pruned the raw list into a smaller set of unique concepts or themes, along with information about which original terms each concept subsumed and how many documents or chunks refer to it.

### Stage 4: Re-Ranking and Core Concept Selection

✅ **Status:** implemented.

We rank the consolidated concepts to decide which are the “core” ones to highlight. Rather than using a simple frequency count, we consider richer metrics:

- **Document frequency / coverage:** Count how many distinct documents or chunks mention the concept. Concepts appearing across many documents indicate broad themes and should rank highly.
- **Relevance score:** Incorporate scores from Stage 2. For example, combine KeyBERT similarity scores across occurrences or leverage topic modeling strength to measure how central a concept is within relevant documents.
- **Manual boosts / domain knowledge:** Boost concepts that match glossary entries or known domain terms. This injects domain knowledge into the ranking.

Using a combination of these signals, we now compute a composite importance score for each cluster via `rank_concept_clusters`. The ranking engine blends Stage 2 scores with document/chunk coverage and occurrence bonuses (`KEYWORD_STAGE4_*` weights) and flags the top `core_limit` results as “core” concepts. The FastAPI endpoint converts those ranked concepts into the JSON payload consumed by the UI, marking the source as `stage4`. If the chat backend is enabled, the list still flows through Stage 5’s keyword filter; otherwise, the Stage 4 output is returned directly. Frequency keywords remain only as a safety fallback.

Ranked concepts also carry a `generated` flag derived from their member phrases, so the UI can indicate when an item ultimately came from the LLM rather than just inspecting the label source.

Because we increased `KEYWORD_STAGE4_SCORE_WEIGHT` to 1.4, Stage 2’s semantic score has more influence—helpful when a concept appears in only a handful of documents.

### Stage 5: Canonical Label Harmonisation (New)

✅ **Status:** implemented.

Even with LLM-assisted cluster labelling, some outputs still include sentiment qualifiers or project-specific suffixes (예: “화주 부정 리뷰”, “화주 긍정 리뷰”). After Stage 4 has ranked the concepts, we introduce a lightweight post-processing pass that asks the chat backend to suggest a single canonical name and brief description for each top-ranked concept. This layer:

- Receives the Stage 4 ranked list with associated aliases/members and prompts the model to produce neutral, semantically representative labels.
- Preserves the original aliases so users can still search or filter by the earlier terms.
- Only runs when an LLM backend is enabled; otherwise, Stage 4 names are returned as-is.

This harmonisation step ensures the UI presents concise, domain-appropriate concept names while keeping traceability via aliases and debug payloads.

### Stage 6: LLM-Based Refinement (Optional)

✅ **Status:** implemented.

Finally, we run a lightweight LLM verification (the existing `KeywordLLMFilter`) over the Stage 5 harmonised list. In verification mode (`KEYWORD_FILTER_ALLOW_GENERATED=false`, the default), the filter can only re-rank or drop items—no new keywords are introduced. This gives stakeholders an interpretable chance to veto noise while preventing the model from hallucinating extra terms. If the flag is enabled, the LLM may append additional concepts, but this should be limited to well-understood scenarios. Deployments without a chat backend simply skip this pass and trust the deterministic ranking from Stages 4–5.

This refinement uses the same OpenAI or Ollama backend and prompting framework already in place. The LLM’s job becomes easier—rather than sifting through dozens of raw tokens, it reviews a handful of refined concepts—which should improve precision.

### Stage 7: Insight Summarisation & Reporting (New)

✅ **Status:** implemented.

After Stage 6 confirms the final keyword list, an optional reporting layer now distils the highest-priority concepts (default: top 12) into up to three analyst-ready insights. The `KeywordLLMFilter.summarize_keywords` prompt captures:

- Concise titles and summaries that explain why the concept cluster matters.
- Optional recommended actions, priority tags, and evidence snippets referencing the contributing keywords.
- Debug instrumentation (`insight_summary`) for every call (status, rejected entries, raw response) so malformed payloads remain debuggable.

The FastAPI route surfaces these insights via a new `insights` field in the JSON payload and records the LLM diagnostics under `stage7` inside the debug block. Operators can toggle the behaviour with `KEYWORD_STAGE7_SUMMARY_ENABLED`, while `KEYWORD_STAGE7_SUMMARY_MAX_CONCEPTS` (defaults to 12) and `KEYWORD_STAGE7_SUMMARY_MAX_INSIGHTS` govern prompt size and output length. Raise `…_MAX_CONCEPTS` if you want the summary to consider more of the long-tail concepts.

If either limit is set to `0`, or the keyword list exceeds `KEYWORD_STAGE7_SUMMARY_MAX_CONCEPTS * 4`, Stage 7 is skipped automatically and the debug payload records the reason (`filter.stage7.reason`).

## Techniques and Algorithms Utilized

The improved pipeline mixes statistical methods, embedding-based methods, and LLM reasoning, playing to each of their strengths. Key techniques include:

- **KeyBERT (embedding-based keyphrase extraction):** Uses embeddings and cosine similarity to find phrases most representative of a document. It captures multi-word keyphrases that may appear only once but have high semantic relevance.
- **RAKE and other statistical extraction methods:** Score phrases by co-occurrence and exclusivity within a document, capturing domain-specific collocations and terms composed of rarely seen words. YAKE or TF-IDF variations can provide similar benefits.
- **Topic modeling (LDA, NMF, BERTopic):** Provides a global view of the corpus by clustering terms into themes. Top topic terms reveal major categories such as “Service Outage & Recovery.”
- **LLM summarization and distillation:** Extracts concepts that are not literal keywords, naming overarching ideas described across documents. Used sparingly, it surfaces implicit concepts.
- **Embedding clustering for synonyms:** Groups similar phrases via cosine similarity thresholds to merge variants (e.g., “login error” vs. “authentication failure”).
- **Re-ranking with document frequency:** Counts the number of payloads or documents associated with each candidate, adding robustness against single-document outliers.

All of these techniques are supported by the current Cornerstone tech stack; the improvement lies in orchestrating the existing pieces more effectively.

## Implementation Plan and Integration

Implementing this pipeline in Cornerstone involves enhancing or adding a few modules:

- **Extend `extract_keyword_candidates` or add a new extraction function:** (✅ done) `extract_concept_candidates` in `keywords.py` performs Stage 2 by blending embeddings, statistical scoring, and optional LLM summaries while emitting rich diagnostics.
- **Aggregate and cluster candidates:** (✅ done) `cluster_concepts` merges candidates using lexical overlap plus cosine similarity from the embedding service, averaging vectors per cluster. Optional LLM labeling (driven by `KEYWORD_STAGE3_LABEL_CLUSTERS`) rewrites cluster names and adds descriptions for the top groups.
- **Implement re-ranking logic:** (✅ done) `rank_concept_clusters` scores each consolidated concept with document/chunk coverage, Stage 2 scores, occurrence bonuses, and label provenance, marking the top results as “core.”
- **Update the API and UI:** (✅ done) `/keywords/{project_id}/candidates` now invokes the multi-stage pipeline and surfaces debug payloads, while the UI consumes harmonised concept lists with progress indicators.
- **Reuse `KeywordLLMFilter`:** (✅ done) Verification prompts review harmonised concepts, respecting `KEYWORD_FILTER_ALLOW_GENERATED` to control whether the model may invent fresh keywords.
- **Testing and iteration:** Exercise the new pipeline on projects of varying sizes to ensure the algorithm alone produces reasonable concepts (especially when the LLM is disabled). Use configuration options for parameters such as max n-gram length, top phrases per document, and cluster thresholds to support easy tuning.

Throughout implementation, maintainability remains a focus by modularizing steps, enabling unit tests for each subroutine, and leveraging existing frameworks. If clustering becomes expensive in real time, it can be offloaded to a background task that updates a cached keyword index on ingestion. Documentation should be updated so future contributors understand how the improved Keyword Explorer operates and where to find each stage’s logic.

## Conclusion

This redesign shifts the Keyword Explorer from a rudimentary word count tool into a robust concept mining pipeline. By integrating embedding-based keyphrase extraction, topic modeling, and selective LLM reasoning, the system now surfaces the core ideas and themes in a knowledge base—not just the frequent words. The pipeline remains compatible with Cornerstone’s existing OpenAI/Ollama stack and leverages those AI capabilities more effectively. The outcome for users is a more meaningful set of keywords: fewer noisy one-word tokens and more informative phrases that echo the business domain. Internally, the modular design and use of proven algorithms make the system easier to maintain and adapt, fulfilling the goals of resilience and ease of use.

With the seven-stage flow implemented end-to-end, upcoming work centres on hardening the reporting layer (stress-testing Ollama-driven runs, tuning prompt length limits) and exploring richer dashboards that combine keyword insights with quantitative metrics. These refinements will keep the Explorer fast, reliable, and explainable as project datasets grow.

## Sources

- Cornerstone current implementation (keyword extraction code and usage)
- KeyBERT project description (BERT-based keyphrase extraction)
- Research on keyword extraction methods (comparing frequency, embeddings, LLMs)
- `keywords.py`: https://github.com/Tocktock/Cornerstone/blob/c62f2e3b1ba37df775d8f37d8289a41fb68a8cad/src/cornerstone/keywords.py
- From Precision to Perception: User-Centred Evaluation of Keyword Extraction Algorithms for Internet-Scale Contextual Advertising: https://arxiv.org/html/2504.21667v1
- `app.py`: https://github.com/Tocktock/Cornerstone/blob/c62f2e3b1ba37df775d8f37d8289a41fb68a8cad/src/cornerstone/app.py
- `cornerstone-improvement-roadmap.md`: https://github.com/Tocktock/Cornerstone/blob/c62f2e3b1ba37df775d8f37d8289a41fb68a8cad/docs/cornerstone-improvement-roadmap.md
- KeyBERT: https://maartengr.github.io/KeyBERT/
- `embeddings.py`: https://github.com/Tocktock/Cornerstone/blob/c62f2e3b1ba37df775d8f37d8289a41fb68a8cad/src/cornerstone/embeddings.py
- `glossary.py`: https://github.com/Tocktock/Cornerstone/blob/c62f2e3b1ba37df775d8f37d8289a41fb68a8cad/src/cornerstone/glossary.py
- `keywords.html`: https://github.com/Tocktock/Cornerstone/blob/c62f2e3b1ba37df775d8f37d8289a41fb68a8cad/templates/keywords.html
