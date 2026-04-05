import { FormEvent, useEffect, useState } from 'react'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatCard } from '../components/StatCard'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { Answer, Decision, Relation, Stats } from '../types/api'

type DashboardPageProps = {
  contextSpaceId: string | null
}

export function DashboardPage({ contextSpaceId }: DashboardPageProps) {
  const [query, setQuery] = useState('Cornerstone')
  const [answer, setAnswer] = useState<Answer | null>(null)

  const stats = useAsyncData<Stats>(
    () => (contextSpaceId ? apiGet('/stats', { context_space_id: contextSpaceId }) : Promise.resolve({ concept_count: 0, relation_count: 0, decision_count: 0, artifact_count: 0, evidence_count: 0 })),
    [contextSpaceId],
  )
  const decisions = useAsyncData<Decision[]>(
    () => (contextSpaceId ? apiGet('/decisions', { context_space_id: contextSpaceId }) : Promise.resolve([])),
    [contextSpaceId],
  )
  const relations = useAsyncData<Relation[]>(
    () => (contextSpaceId ? apiGet('/relations', { context_space_id: contextSpaceId }) : Promise.resolve([])),
    [contextSpaceId],
  )

  useEffect(() => {
    if (!contextSpaceId) {
      return
    }
    apiGet<Answer>('/answers', { q: query, context_space_id: contextSpaceId }).then(setAnswer).catch(() => setAnswer(null))
  }, [contextSpaceId])

  async function onSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!contextSpaceId) {
      return
    }
    const result = await apiGet<Answer>('/answers', { q: query, context_space_id: contextSpaceId })
    setAnswer(result)
  }

  if (!contextSpaceId) {
    return <EmptyState title="No context space yet" description="The backend is still preparing demo data." />
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Operational overview"
        description="Track the curated context layer across artifacts, glossary, graph, and decision records."
      />

      <section className="stats-grid">
        <StatCard label="Concepts" value={stats.data?.concept_count ?? 0} helper="Curated glossary entries" />
        <StatCard label="Relations" value={stats.data?.relation_count ?? 0} helper="Official ontology edges" />
        <StatCard label="Decisions" value={stats.data?.decision_count ?? 0} helper="Reviewable decision context" />
        <StatCard label="Evidence" value={stats.data?.evidence_count ?? 0} helper="Source-backed supporting fragments" />
      </section>

      <section className="two-column-layout">
        <article className="panel search-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Structured answer</span>
              <h3>Ask the context layer</h3>
            </div>
          </div>

          <form className="search-form" onSubmit={onSearch}>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask about a term or decision" />
            <button type="submit">Search</button>
          </form>

          {answer ? (
            <div className="answer-block">
              <p className="answer-summary">{answer.summary}</p>
              <div className="chip-row">
                {answer.concepts.map((concept) => (
                  <span key={concept.id} className="chip">
                    {concept.canonical_name}
                  </span>
                ))}
              </div>
              <div className="evidence-list compact-list">
                {answer.evidence.slice(0, 4).map((item) => (
                  <article key={item.id} className="evidence-card">
                    <div className="card-row between">
                      <strong>{item.artifact_title}</strong>
                      <StatusPill value={item.verification_status} />
                    </div>
                    <p>{item.excerpt}</p>
                    <a href={item.artifact_url} target="_blank" rel="noreferrer">
                      Open source
                    </a>
                  </article>
                ))}
              </div>
            </div>
          ) : (
            <p className="muted">Run a query to preview source-backed answers.</p>
          )}
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Recent decisions</span>
              <h3>Decision context</h3>
            </div>
          </div>
          <div className="stack-list">
            {(decisions.data ?? []).slice(0, 4).map((decision) => (
              <article key={decision.id} className="list-card">
                <div className="card-row between">
                  <strong>{decision.title}</strong>
                  <StatusPill value={decision.status} />
                </div>
                <p>{decision.decision}</p>
              </article>
            ))}
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Graph snapshot</span>
            <h3>Key official relations</h3>
          </div>
        </div>
        <div className="stack-list">
          {(relations.data ?? []).slice(0, 6).map((relation) => (
            <article key={relation.id} className="list-card relation-card">
              <div className="relation-line">
                <span>{relation.subject_name}</span>
                <span className="relation-predicate">{relation.predicate}</span>
                <span>{relation.object_name}</span>
              </div>
              <p>{relation.description}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
