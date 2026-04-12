import { FormEvent, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ArtifactCard, AlertBanner, ProvenanceStrip, SectionIntro } from '../components/experience'
import { PageHeader } from '../components/PageHeader'
import { StatCard } from '../components/StatCard'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type {
  ActorSession,
  AnswerPayload,
  ContractEnvelope,
  NoMatchPayload,
  RuntimeBootstrapMeta,
  SearchResultItem,
  SearchResultsPayload,
  WorkspaceHomePayload,
} from '../types/api'
import {
  canActorManageConnectors,
  formatLocalDateTime,
  isProductionWorkspaceDegraded,
  isProductionWorkspacePending,
} from '../viewModels'

type WorkspacePageProps = {
  activeActor: ActorSession
  runtimeInfo: RuntimeBootstrapMeta
}

export function WorkspacePage({ activeActor, runtimeInfo }: WorkspacePageProps) {
  const home = useAsyncData<ContractEnvelope<WorkspaceHomePayload>>(
    () => apiGet('/workspace-home'),
    [activeActor.actor_id],
  )
  const [query, setQuery] = useState('escalation')
  const [answer, setAnswer] = useState<ContractEnvelope<AnswerPayload | NoMatchPayload> | null>(null)
  const [searchResults, setSearchResults] = useState<ContractEnvelope<SearchResultsPayload | NoMatchPayload> | null>(
    null,
  )
  const [searchError, setSearchError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const activeAnswer = answer ?? home.data?.payload.featured_answer ?? null
  const featuredCards = home.data?.payload.featured_cards ?? []
  const recentChanges = home.data?.payload.recent_changes ?? []
  const freshnessAlerts = home.data?.payload.freshness_alerts ?? []
  const reviewSummary = home.data?.payload.review_queue_summary
  const sourceSummary = home.data?.payload.source_health_summary
  const leadFeaturedCard = featuredCards[0] ?? null
  const supportingFeaturedCards = leadFeaturedCard ? featuredCards.slice(1) : []
  const canManageConnectors = canActorManageConnectors(activeActor)
  const productionPending = isProductionWorkspacePending(runtimeInfo)
  const productionDegraded = isProductionWorkspaceDegraded(runtimeInfo)
  const sourceStudioAction = canManageConnectors ? (
    <Link className="ghost-link" to="/source-studio">
      Open Source Studio
    </Link>
  ) : null

  const searchHeading = useMemo(() => {
    if (!activeAnswer) {
      return 'Featured answer'
    }
    if (isAnswerPayload(activeAnswer.payload)) {
      return query
    }
    return activeAnswer.payload.reason
  }, [activeAnswer, query])

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
      setSearchError(error instanceof Error ? error.message : 'Workspace query failed.')
    } finally {
      setIsSubmitting(false)
    }
  }

  if (home.error) {
    return <EmptyState title="Workspace unavailable" description={home.error} />
  }

  if (productionPending) {
    const awaitingSources = runtimeInfo.workspace_data_state === 'awaiting_sources'
    return (
      <section className="page-stack reader-page workspace-page">
        <PageHeader
          eyebrow="Cornerstone workspace"
          title="Workspace"
        />
        {awaitingSources ? (
          <EmptyState
            eyebrow="Production onboarding"
            title="Connect a shared datasource first"
            description={
              canManageConnectors
                ? 'Production workspaces do not fall back to demo content. Connect a shared datasource in Source Studio.'
                : 'This production workspace has no shared datasource yet. A connector manager needs to connect one first.'
            }
            actions={sourceStudioAction}
          />
        ) : (
          <EmptyState
            eyebrow="Production sync"
            title="Sources connected, first sync in progress"
            description={`${runtimeInfo.linked_source_count} linked sources are preparing the first usable workspace artifacts.`}
            actions={sourceStudioAction}
          />
        )}
      </section>
    )
  }

  return (
    <section className="page-stack reader-page workspace-page">
      <PageHeader
        eyebrow="Cornerstone workspace"
        title="Workspace"
      />

      {searchError ? <AlertBanner tone="danger" title="Workspace query failed" description={searchError} /> : null}
      {productionDegraded ? (
        <AlertBanner
          tone="danger"
          eyebrow="Production recovery"
          title="Source recovery cues remain active"
          description={`${runtimeInfo.degraded_source_count} linked sources currently need recovery attention.`}
        />
      ) : null}

      <div className="workspace-stage-grid">
        <article className="hero-block workspace-hero">
          <div className="hero-copy">
            <span className="eyebrow">Workspace home</span>
            <h2>Ask the shared context layer first.</h2>
            <p>{home.data?.payload.hero_prompt ?? 'Ask about official workspace context and decision lineage.'}</p>
          </div>

          <form className="hero-form" onSubmit={handleQuery}>
            <label className="hero-input">
              <span className="mini-label">Ask Cornerstone</span>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search workspace context"
              />
            </label>
            <button type="submit" disabled={isSubmitting}>
              {isSubmitting ? 'Searching…' : 'Run query'}
            </button>
          </form>

          {leadFeaturedCard ? (
            <div className="hero-support-card">
              <span className="mini-label">Current featured artifact</span>
              <strong>{leadFeaturedCard.title}</strong>
              <p>{leadFeaturedCard.summary}</p>
              <div className="artifact-status-row">
                <StatusPill value={leadFeaturedCard.support_visibility} />
                {leadFeaturedCard.verification_state ? <StatusPill value={leadFeaturedCard.verification_state} /> : null}
              </div>
            </div>
          ) : null}
        </article>

        <article className="reader-primary-panel workspace-answer-panel">
          <SectionIntro
            eyebrow="Featured answer"
            title={searchHeading}
            actions={activeAnswer ? <StatusPill value={activeAnswer.response_kind} /> : null}
          />

          {activeAnswer ? (
            isAnswerPayload(activeAnswer.payload) ? (
              <>
                <p className="artifact-lead">{activeAnswer.payload.answer_text}</p>
                <ProvenanceStrip
                  summary={activeAnswer.payload.provenance_summary}
                  supportVisibility={activeAnswer.payload.support_visibility}
                  verificationState={activeAnswer.payload.verification_state}
                  variant="rail"
                />
                <div className="narrative-grid answer-sections-grid">
                  {activeAnswer.payload.answer_sections.map((section) => (
                    <article key={section.heading} className="narrative-section narrative-section-panel">
                      <h4>{section.heading}</h4>
                      <p>{section.body}</p>
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <EmptyState
                title={activeAnswer.payload.reason}
                description={activeAnswer.payload.request_rewrite_hint ?? 'No answer available for the current prompt.'}
                eyebrow="Official answer state"
              />
            )
          ) : (
            <EmptyState
              title="Workspace home is loading"
              description="The featured answer will render here."
              eyebrow="Answer state"
            />
          )}
        </article>
      </div>

      <div className="summary-strip workspace-summary-strip">
        {reviewSummary ? (
          <StatCard
            label="Review cues"
            value={reviewSummary.pending_count}
            helper={`${reviewSummary.officialize_ready_count} ready to officialize · ${reviewSummary.review_required_count} need review attention.`}
            tone="info"
          />
        ) : null}
        {sourceSummary ? (
          <StatCard
            label="Source health"
            value={sourceSummary.total_count}
            helper={`${sourceSummary.active_count} active · ${sourceSummary.degraded_count} degraded · ${sourceSummary.stale_count} stale.`}
            tone={sourceSummary.degraded_count || sourceSummary.stale_count ? 'danger' : 'sage'}
          />
        ) : null}
      </div>

      <div className="reader-two-column workspace-content-grid">
        <section className="reader-secondary-panel workspace-river-panel">
          <SectionIntro
            eyebrow="Recent changes"
            title="Freshly updated workspace artifacts"
          />
          <div className="workspace-river">
            {recentChanges.map((item, index) => (
              <article
                key={`${item.resource_ref.resource_kind}-${item.resource_ref.resource_id}`}
                className={`artifact-card artifact-card-${index === 0 ? 'lead' : 'rail'} river-card`}
              >
                <div className="artifact-card-header">
                  <div>
                    <span className="eyebrow">{item.resource_ref.resource_kind}</span>
                    <h3>{item.resource_ref.resource_label}</h3>
                  </div>
                  <Link
                    className="artifact-link"
                    to={item.resource_ref.resource_kind === 'concept' ? `/concepts/${item.public_slug}` : `/decisions/${item.public_slug}`}
                  >
                    Read
                  </Link>
                </div>
                <div className="artifact-status-row">
                  <StatusPill value={item.lifecycle_state} />
                  <StatusPill value={item.support_visibility} />
                  {item.verification_state ? <StatusPill value={item.verification_state} /> : null}
                </div>
                <p>{item.change_summary}</p>
                <p className="meta-copy">Updated {formatLocalDateTime(item.changed_at)}</p>
              </article>
            ))}
          </div>
        </section>

        <aside className="reader-secondary-panel workspace-support-rail">
          <SectionIntro
            eyebrow={searchResults && isSearchResultsPayload(searchResults.payload) ? 'Search results' : 'Featured references'}
            title={
              searchResults && isSearchResultsPayload(searchResults.payload)
                ? `${searchResults.payload.result_count} matches`
                : 'Explore references'
            }
          />

          {searchResults && isSearchResultsPayload(searchResults.payload) ? (
            <ul className="stack-list">
              {searchResults.payload.results.map((result) => (
                <li key={result.resource_ref.resource_id} className="list-card compact-card search-result-card">
                  <div className="card-row between start">
                    <strong>{result.resource_ref.resource_label}</strong>
                    {resolveSearchHref(result) ? (
                      <Link className="artifact-link" to={resolveSearchHref(result) as string}>
                        Open
                      </Link>
                    ) : null}
                  </div>
                  <p>{result.match_reason_summary}</p>
                  <div className="artifact-status-row">
                    {result.support_visibility ? <StatusPill value={result.support_visibility} /> : null}
                    {result.verification_state ? <StatusPill value={result.verification_state} /> : null}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="stack-list">
              {(leadFeaturedCard ? [leadFeaturedCard, ...supportingFeaturedCards] : featuredCards).map((card, index) => (
                <ArtifactCard
                  key={`${card.resource_ref.resource_kind}-${card.resource_ref.resource_id}`}
                  to={card.resource_ref.resource_kind === 'concept' ? `/concepts/${card.public_slug}` : `/decisions/${card.public_slug}`}
                  eyebrow={card.eyebrow}
                  title={card.title}
                  summary={card.summary}
                  supportVisibility={card.support_visibility}
                  lifecycleState={card.lifecycle_state}
                  verificationState={card.verification_state}
                  variant={index === 0 ? 'standard' : 'compact'}
                  ctaLabel="Read"
                />
              ))}
            </div>
          )}

          {freshnessAlerts.length ? (
            <section className="narrative-section compact-section">
              <SectionIntro
                eyebrow="Freshness alerts"
                title="Quiet operational cues"
                compact
              />
              <div className="alert-grid compact-alert-grid">
                {freshnessAlerts.map((alert) => (
                  <AlertBanner
                    key={alert.source_connection_id}
                    tone={alert.source_connection_state === 'degraded' || alert.freshness_state === 'stale' ? 'danger' : 'default'}
                    eyebrow={alert.freshness_state}
                    title={`${alert.source_label} · ${alert.source_connection_state}`}
                    description={`${alert.note} Last success: ${formatLocalDateTime(alert.last_successful_sync_at)}.`}
                  />
                ))}
              </div>
            </section>
          ) : null}
        </aside>
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

function resolveSearchHref(result: SearchResultItem) {
  if (result.resource_ref.resource_kind === 'concept') {
    return `/explore/topics`
  }
  if (result.resource_ref.resource_kind === 'decision') {
    return `/explore/decisions`
  }
  return '/explore/map'
}
