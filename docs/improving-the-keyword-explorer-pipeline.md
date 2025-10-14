# Improving the Keyword Explorer Pipeline

## Current Implementation and Its Limitations

Following the first wave of improvements, the Keyword Explorer now performs the Stage 1 chunk preparation and Stage 2 concept extraction described in this document. Incoming payloads are normalised into `KeywordSourceChunk` objects, and each chunk is analysed with a hybrid pipeline (n‑gram scoring, RAKE-style statistics, embedding similarity, and optional LLM summaries) that produces rich `ConceptCandidate` entries complete with coverage metrics and score breakdowns. Stage 2 has extensive instrumentation (`keyword.stage2.*` logs, debug payloads, and per-component timing) and is fully configurable through the new `KEYWORD_STAGE2_*` environment variables.

Despite these upgrades, the UI still displays the legacy frequency list because Stage 3 (concept consolidation), Stage 4 (re-ranking), and the revised Stage 5 LLM review are not implemented yet. As a result:

- We continue to show single-token keywords by default, even though multi-word concepts are now detected in Stage 2.
- The LLM refine step currently targets the frequency list; malformed responses still trigger fallback warnings (“missing-keywords-array”).
- Concept clustering relies on a lightweight token-overlap heuristic and is only surfaced in debug output.

The remaining stages in this document describe the work required to replace that frequency fallback with the new semantic concepts.

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
5. LLM-based refinement (optional)

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

Remaining Stage 2 follow-ups:

- Surface the stored phrase embeddings and contribution metadata so Stage 3 can perform true embedding-based clustering without recomputing vectors.
- Expand automated tests to cover malformed LLM responses (currently handled defensively but not asserted).

### Stage 3: Concept Consolidation and Clustering

✅ **Status:** implemented (embedding-aware clustering with optional LLM labeling).

After gathering candidates, we often end up with semantically overlapping terms (e.g., “login issue,” “sign-in problem”) or different granularities (specific error codes vs. broader categories). This stage groups and condenses the candidate list into core concepts:

- **Clustering by semantic similarity:** Compute embeddings for each candidate phrase (we may reuse those from the KeyBERT step) and cluster them (e.g., with K-means, HDBSCAN, or a simple similarity threshold). This groups near-duplicates or synonyms, ensuring we do not present multiple variants of the same concept.
- **Topic modeling:** Run a topic modeling approach (such as LDA, NMF, or BERTopic) across all documents to discover themes. Topic representatives like “Payment Issues” or “Account Management” provide high-level context. This step can be run offline to guide clustering.
- **Consolidation into representative concepts:** For each cluster of similar candidates, choose a representative term or phrase. In the current implementation we pick the highest-scoring member by default and, when `KEYWORD_STAGE3_LABEL_CLUSTERS=true`, ask the configured chat backend to suggest a concise cluster label and optional description via `KeywordLLMFilter.label_clusters`. Results surface aliases, label provenance, and debug payloads for traceability. Disable or cap labeling with `KEYWORD_STAGE3_LABEL_MAX_CLUSTERS`.

At the end of Stage 3, we have pruned the raw list into a smaller set of unique concepts or themes, along with information about which original terms each concept subsumed and how many documents or chunks refer to it.

### Stage 4: Re-Ranking and Core Concept Selection

We rank the consolidated concepts to decide which are the “core” ones to highlight. Rather than using a simple frequency count, we consider richer metrics:

- **Document frequency / coverage:** Count how many distinct documents or chunks mention the concept. Concepts appearing across many documents indicate broad themes and should rank highly.
- **Relevance score:** Incorporate scores from Stage 2. For example, combine KeyBERT similarity scores across occurrences or leverage topic modeling strength to measure how central a concept is within relevant documents.
- **Manual boosts / domain knowledge:** Boost concepts that match glossary entries or known domain terms. This injects domain knowledge into the ranking.

Using a combination of these signals, compute a composite importance score for each concept, sort, and select the top concepts as the final core keywords to present. This multi-factor ranking is more robust than raw token frequency and naturally highlights higher-level themes.

### Stage 5: LLM-Based Refinement (Optional)

Finally, we can include a lighter LLM review similar to the current system’s `KeywordLLMFilter`, but with a different role. At this point, we have a curated list of top concepts. We can prompt the LLM to verify that the concepts are correct, distinct, and meaningful. The model might flag generic entries or suggest missing concepts. Any adjustments are applied to finalize the list. If no LLM is available, we simply skip this step; the Stage 4 output already offers high-quality results.

This refinement uses the same OpenAI or Ollama backend and prompting framework already in place. The LLM’s job becomes easier—rather than sifting through dozens of raw tokens, it reviews a handful of refined concepts—which should improve precision.

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

- **Extend `extract_keyword_candidates` or add a new extraction function:** Create an `extract_concept_candidates` function in `keywords.py` that performs Stage 2. Iterate over documents or chunks, call the embedding service for vectors, compute cosine similarities, run RAKE (or similar), and return candidates with additional metrics such as scores and originating document IDs.
- **Aggregate and cluster candidates:** (✅ done) `cluster_concepts` merges candidates using lexical overlap plus cosine similarity from the embedding service, averaging vectors per cluster. Optional LLM labeling (driven by `KEYWORD_STAGE3_LABEL_CLUSTERS`) rewrites cluster names and adds descriptions for the top groups.
- **Implement re-ranking logic:** Add a function to score each consolidated concept, leveraging document frequency, relevance metrics, and optional topic weights. Output a sorted list of `KeywordCandidate` objects, marking `is_core=True` for the top results and redefining the `count` field to store a meaningful metric (such as document frequency).
- **Update the API and UI:** Adjust `/keywords/{project_id}/candidates` in `app.py` to call the new pipeline and optionally run the existing `KeywordLLMFilter` as a final review. Keep the JSON schema compatible with the frontend, including fields like `term`, `count`, `core`, `generated`, `reason`, and `source`. Extend the debug payload with information about the new stages.
- **Reuse `KeywordLLMFilter`:** Retune its prompt for the verification role, passing synthesized context rather than raw snippets. Minimal code changes are required aside from prompt adjustments and parsing any new flags.
- **Testing and iteration:** Exercise the new pipeline on projects of varying sizes to ensure the algorithm alone produces reasonable concepts (especially when the LLM is disabled). Use configuration options for parameters such as max n-gram length, top phrases per document, and cluster thresholds to support easy tuning.

Throughout implementation, maintainability remains a focus by modularizing steps, enabling unit tests for each subroutine, and leveraging existing frameworks. If clustering becomes expensive in real time, it can be offloaded to a background task that updates a cached keyword index on ingestion. Documentation should be updated so future contributors understand how the improved Keyword Explorer operates and where to find each stage’s logic.

## Conclusion

This redesign shifts the Keyword Explorer from a rudimentary word count tool into a robust concept mining pipeline. By integrating embedding-based keyphrase extraction, topic modeling, and selective LLM reasoning, the system will surface the core ideas and themes in a knowledge base—not just the frequent words. The pipeline remains compatible with Cornerstone’s existing OpenAI/Ollama stack and leverages those AI capabilities more effectively. The outcome for users will be a more meaningful set of keywords: fewer noisy one-word tokens and more informative phrases that echo the business domain. Internally, the modular design and use of proven algorithms make the system easier to maintain and adapt, fulfilling the goals of resilience and ease of use.

By addressing the identified weaknesses—chiefly, moving beyond shallow frequency counts—the new pipeline ensures that Cornerstone’s Keyword Explorer captures deeper domain-level concepts and provides high-level insights that were previously missing. The next steps involve implementing these changes in the codebase, iterating with real data, and delivering a far richer keyword exploration experience.

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
