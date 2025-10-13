# Improving the Keyword Explorer Pipeline

## Current Implementation and Its Limitations

The Cornerstone project’s “Keyword Explorer” currently uses a straightforward frequency-based approach to extract keywords from documents. All text chunks in a project are scanned, and every token (continuous sequence of letters or Hangul characters) is lowercased and counted. Common stopwords, very short tokens, and purely numeric strings are filtered out. The result is a list of single-word `KeywordCandidate` entries, each with a raw frequency count. The top frequent terms are marked as “core” keywords if they appear more than once. This method is simple but shallow—it treats individual tokens as keywords and relies on raw occurrence counts for importance.

Such a frequency-driven technique has clear weaknesses. Multi-word concepts are split up and lost—for example, a domain term like “login issue” would be counted as “login” and “issue” separately, missing the combined concept. The approach also tends to surface generic frequent words rather than true domain themes. A common word like “error” or “system” might rank high by frequency but conveys little about the specific domain or business context. Conversely, an important concept that appears only once or twice, such as the name of a product feature or a rare technical term, would be ignored entirely by a frequency cutoff. In short, raw counts do not always correlate with semantic importance, especially in a specialized corpus.

Currently, an LLM-based filter is layered on top to mitigate noise. If an OpenAI or Ollama backend is configured, the code passes the frequency candidates through a `KeywordLLMFilter`. This filter prompts a language model with the list of candidate words and a few text snippets as context, asking it to identify meaningful product names, process steps, issue categories, and other high-value concepts. The LLM returns a JSON list of keywords to keep (or new ones to add) along with brief reasons. The system then marks kept terms as core and may include a few generated keywords that the LLM introduced (`source="generated"`). This adds a semantic layer, filtering out trivial tokens like file names or IDs and occasionally adding missing concepts from the context.

However, this LLM refinement stage has its own limitations. It only sees a limited window of context (by default the first five snippets up to 400 characters each), which might not cover all topics in the corpus. It also returns at most ten keywords to avoid long outputs, potentially omitting additional relevant concepts if the knowledge base is broad. If the LLM is disabled or fails, the system falls back to the raw frequency list—which, as noted, may be noisy or incomplete. The current pipeline often fails to capture deeper domain-level concepts. It leans heavily on counting words, which misses multi-word phrases and nuance, and it relies on a single LLM prompt to retroactively inject semantics, which is a fragile solution.

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

In this stage, we ensure the documents are segmented in a way that preserves context for concept extraction. Cornerstone already splits files into chunks during ingestion (with plans for hierarchical splitting by headings). We will leverage these embedded chunks as the unit of analysis. Each chunk or document is a moderate-sized text (for example, 200–500 tokens) that can be processed for keywords. Using chunks rather than the entire concatenated corpus prevents extremely long text from diluting important concepts and allows capturing context-specific phrases.

For each chunk, we retrieve the raw text (already stored in the Qdrant payloads). We may optionally normalize the text (removing punctuation, lowercasing for English, etc.), but we will preserve multi-word sequences. Instead of splitting on every whitespace, we can keep potential phrases intact for analysis. For example, we might use a phrase tokenizer or consider sequences of words (up to two or three words) as candidates in the next stage. The existing regex in `extract_keyword_candidates` can be extended or replaced to capture n-grams. Currently it only captures single tokens; we could post-process the token list to join frequent adjacent tokens or use an external phrase extraction algorithm. The chunking stage also allows capturing document-level metadata (titles, headings) that might hint at key concepts, which we can feed into later stages.

### Stage 2: Candidate Concept Extraction

This is a critical new stage where we generate an initial pool of candidate keywords and key phrases from each chunk. We will combine multiple techniques to gather a rich candidate set:

- **Embedding-based key phrase extraction (KeyBERT-like):**
  - Compute the chunk’s embedding vector using the existing `EmbeddingService`, which supports OpenAI or local models.
  - Generate candidate n-grams from the chunk text (e.g., all one- to three-word sequences, perhaps filtered by part-of-speech to focus on noun phrases).
  - Compute embeddings for each candidate phrase.
  - Rank phrases by cosine similarity to the chunk embedding and take the top phrases from each chunk as candidates (for example, the top three phrases). This ensures even if a concept appears only in one document, it can still surface via that document’s top phrases.

- **Statistical keyword extraction (RAKE, YAKE, or TF-IDF):**
  - Run a traditional algorithm such as Rapid Automatic Keyword Extraction (RAKE) or YAKE on each chunk to get a different perspective.
  - These methods highlight phrases that appear several times within the chunk or contain rare informative words, catching domain-specific collocations (e.g., “SLA breach,” “OAuth token”) that might not rank highest by embedding similarity.
  - Alternatively, a TF-IDF scoring across the corpus can flag terms that are frequent in one document but not others, surfacing distinctive concepts.

- **LLM-based extraction (summarization):**
  - For particularly large or complex documents, optionally use the LLM in a summarization role by prompting it to list key points or concepts.
  - The LLM’s answers, expressed in sentences or phrases, reveal important domain concepts in natural language. Parsing those answers yields concept phrases.
  - This method captures high-level business themes that might not be repeated as literal keywords. It can be toggled on or off to control cost.

Each of these techniques produces a list of candidate keywords or phrases (likely with an importance score). We merge the candidates from all chunks; the union represents a broad superset of potential domain concepts. Some redundancy is acceptable at this stage—the next step consolidates the list.

### Stage 3: Concept Consolidation and Clustering

After gathering candidates, we often end up with semantically overlapping terms (e.g., “login issue,” “sign-in problem”) or different granularities (specific error codes vs. broader categories). This stage groups and condenses the candidate list into core concepts:

- **Clustering by semantic similarity:** Compute embeddings for each candidate phrase (we may reuse those from the KeyBERT step) and cluster them (e.g., with K-means, HDBSCAN, or a simple similarity threshold). This groups near-duplicates or synonyms, ensuring we do not present multiple variants of the same concept.
- **Topic modeling:** Run a topic modeling approach (such as LDA, NMF, or BERTopic) across all documents to discover themes. Topic representatives like “Payment Issues” or “Account Management” provide high-level context. This step can be run offline to guide clustering.
- **Consolidation into representative concepts:** For each cluster of similar candidates, choose a representative term or phrase. A heuristic is to pick the candidate with the highest average score or the one appearing in the most documents. Alternatively, use the LLM to label the cluster with an overarching concept name. The result is a set of distinct, high-level concepts.

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
- **Aggregate and cluster candidates:** Compute embeddings for each candidate phrase and cluster them using either simple similarity thresholds or libraries such as scikit-learn or BERTopic. This step could run on-demand or be moved to ingestion to cache results for faster responses.
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

