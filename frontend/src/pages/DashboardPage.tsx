import { FormEvent, useState } from 'react'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatCard } from '../components/StatCard'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type {
  ActorSession,
  AnswerPayload,
  ConceptPayload,
  ContextSpaceRef,
  ContractEnvelope,
  DecisionPayload,
  NoMatchPayload,
  SearchResultsPayload,
  SearchResultItem,
  SourceConnectionStatus,
} from '../types/api'

type DashboardPageProps = {
  workspace: ContextSpaceRef
  activeActor: ActorSession
}

export function DashboardPage({ workspace, activeActor }: DashboardPageProps) {
  const concepts = useAsyncData<ContractEnvelope<ConceptPayload>[]>(
    () => apiGet('/concepts'),
    [activeActor.actor_id],
  )
  const decisions = useAsyncData<ContractEnvelope<DecisionPayload>[]>(
    () => apiGet('/decisions'),
    [activeActor.actor_id],
  )
  const sources = useAsyncData<SourceConnectionStatus[]>(
    () => apiGet('/source-connections'),
    [activeActor.actor_id],
  )
  const [query, setQuery] = useState('escalation')
  const [answer, setAnswer] = useState<ContractEnvelope<AnswerPayload | NoMatchPayload> | null>(null)
  const [searchResults, setSearchResults] = useState<ContractEnvelope<SearchResultsPayload | NoMatchPayload> | null>(
    null,
  )
  const [searchError, setSearchError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setIsSubmitting(true)
    setSearchError(null)
    try {
      const [answerResult, searchResult] = await Promise.all([
        apiGet<ContractEnvelope<AnswerPayload | NoMatchPayload>>('/answers', { q: query }),
        apiGet<ContractEnvelope<SearchResultsPayload | NoMatchPayload>>('/search', { q: query }),
      ])
      setAnswer(answerResult)
      setSearchResults(searchResult)
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : 'Search failed.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <section className="page-stack">
      <PageHeader
        title="Workspace overview"
        description={`Viewing ${workspace.context_space_name} as ${activeActor.display_name}. The dashboard keeps search, trust disclosure, and source health visible in one place.`}
      />

      <div className="stats-grid">
        <StatCard
          label="Concepts"
          value={concepts.data?.length ?? 0}
          helper="Published concepts returned through the canonical envelope."
        />
        <StatCard
          label="Decisions"
          value={decisions.data?.length ?? 0}
          helper="Accepted or superseded decision records visible to this persona."
        />
        <StatCard
          label="Sources"
          value={sources.data?.length ?? 0}
          helper="Operational source connections in the current workspace."
        />
        <StatCard
          label="Restricted"
          value={concepts.data?.filter((item) => item.payload.support_visibility === 'restricted_support').length ?? 0}
          helper="Official outputs that disclose hidden decisive support."
        />
      </div>

      <article className="panel">
        <PageHeader
          title="Search and grounded answers"
          description="This uses the same contract semantics as the MCP-style read path."
        />
        <form className="search-form" onSubmit={handleQuery}>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search workspace context" />
          <button type="submit" disabled={isSubmitting}>
            {isSubmitting ? 'Searching…' : 'Run search'}
          </button>
        </form>

        {searchError ? <EmptyState title="Search failed" description={searchError} /> : null}

        {answer ? (
          <div className="two-column-layout">
            <article className="panel nested-panel">
              <span className="eyebrow">Answer</span>
              <div className="inline-meta">
                <StatusPill value={answer.response_kind} />
                <StatusPill value={isAnswerPayload(answer.payload) ? answer.payload.support_visibility : answer.payload.reason} />
              </div>
              {isAnswerPayload(answer.payload) ? (
                <>
                  <p>{answer.payload.answer_text}</p>
                  <p className="muted">
                    Verification: {answer.payload.verification_state} · Visible support items:{' '}
                    {answer.payload.visible_support_items.length}
                  </p>
                </>
              ) : (
                <EmptyState title={answer.payload.reason} description={answer.payload.request_rewrite_hint ?? 'No answer available.'} />
              )}
            </article>

            <article className="panel nested-panel">
              <span className="eyebrow">Search results</span>
              {searchResults && isSearchResultsPayload(searchResults.payload) ? (
                <ul className="stack-list">
                  {searchResults.payload.results.map((result: SearchResultItem) => (
                    <li key={result.resource_ref.resource_id} className="list-card">
                      <strong>{result.resource_ref.resource_label}</strong>
                      <p>{result.match_reason_summary}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <EmptyState
                  title={searchResults && !isSearchResultsPayload(searchResults.payload) ? searchResults.payload.reason : 'Run a search'}
                  description={
                    searchResults && !isSearchResultsPayload(searchResults.payload)
                      ? (searchResults.payload.request_rewrite_hint ?? 'Search results will appear here.')
                      : 'Search results will appear here.'
                  }
                />
              )}
            </article>
          </div>
        ) : null}
      </article>

      <div className="card-grid">
        {(concepts.data ?? []).slice(0, 3).map((envelope) => (
          <article key={envelope.payload.concept_id} className="panel nested-panel">
            <span className="eyebrow">Concept</span>
            <h3>{envelope.payload.canonical_name}</h3>
            <div className="inline-meta">
              <StatusPill value={envelope.payload.support_visibility} />
              <StatusPill value={envelope.payload.verification_state} />
            </div>
            <p>{envelope.payload.definition}</p>
          </article>
        ))}
      </div>
    </section>
  )
}

function isAnswerPayload(payload: AnswerPayload | NoMatchPayload): payload is AnswerPayload {
  return 'answer_text' in payload
}

function isSearchResultsPayload(payload: SearchResultsPayload | NoMatchPayload): payload is SearchResultsPayload {
  return 'results' in payload
}
