import { FormEvent, useMemo, useState } from 'react'

import { apiDelete, apiGet, apiPatch, apiPost } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type {
  ActorSession,
  ConnectorTemplateSummary,
  ContextSpaceRef,
  ProviderBindingStartResponse,
  ProviderBindingSummary,
  SourceConnectionCreatePayload,
  SourceConnectionDetail,
  SourceConnectionPreviewResponse,
  SourceConnectionStatus,
  SourceConnectionUpdatePayload,
  SyncRunSummary,
} from '../types/api'
import { canActorManageConnectors, formatLocalDateTime } from '../viewModels'

type SourcesPageProps = {
  workspace: ContextSpaceRef
  activeActor: ActorSession
}

type EditorState = {
  connectionId: string
  sourceLabel: string
  selectedScopeInput: string
  visibilityClass: 'member_visible' | 'evidence_only'
  syncIntervalSeconds: number
  providerCredentialRef: string
}

const DEFAULT_TEMPLATE = 'notion_shared_page_tree'

function renderDateTime(value?: string | null, fallback = 'Never') {
  if (!value) {
    return fallback
  }

  return <time dateTime={value}>{formatLocalDateTime(value)}</time>
}

export function SourcesPage({ workspace, activeActor }: SourcesPageProps) {
  const canManage = canActorManageConnectors(activeActor)
  const [refreshIndex, setRefreshIndex] = useState(0)
  const [templateKey, setTemplateKey] = useState(DEFAULT_TEMPLATE)
  const [sourceLabel, setSourceLabel] = useState('')
  const [selectedScopeInput, setSelectedScopeInput] = useState('')
  const [visibilityClass, setVisibilityClass] = useState<'member_visible' | 'evidence_only'>(
    'member_visible',
  )
  const [syncIntervalSeconds, setSyncIntervalSeconds] = useState('900')
  const [binding, setBinding] = useState<ProviderBindingStartResponse | ProviderBindingSummary | null>(null)
  const [authorizationCode, setAuthorizationCode] = useState('')
  const [preview, setPreview] = useState<SourceConnectionPreviewResponse | null>(null)
  const [managerError, setManagerError] = useState<string | null>(null)
  const [managerNotice, setManagerNotice] = useState<string | null>(null)
  const [isBinding, setIsBinding] = useState(false)
  const [isPreviewing, setIsPreviewing] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [editing, setEditing] = useState<EditorState | null>(null)
  const [runsByConnection, setRunsByConnection] = useState<Record<string, SyncRunSummary[]>>({})
  const [expandedRunsId, setExpandedRunsId] = useState<string | null>(null)

  const sources = useAsyncData<SourceConnectionStatus[]>(
    () => apiGet('/source-connections'),
    [workspace.context_space_id, activeActor.actor_id, refreshIndex],
  )
  const templates = useAsyncData<ConnectorTemplateSummary[]>(
    () => (canManage ? apiGet('/connector-templates') : Promise.resolve([])),
    [activeActor.actor_id, canManage],
  )

  const availableTemplates = templates.data ?? []
  const selectedTemplate =
    availableTemplates.find((template) => template.template_key === templateKey) ?? availableTemplates[0]
  const boundCredentialRef =
    binding && 'provider_credential_ref' in binding ? binding.provider_credential_ref : null

  const recommendedInterval = useMemo(() => {
    if (!selectedTemplate) {
      return 900
    }
    return selectedTemplate.recommended_sync_interval_seconds
  }, [selectedTemplate])

  if (sources.error) {
    return <EmptyState title="Source status unavailable" description={sources.error} />
  }

  async function handleStartBinding() {
    setIsBinding(true)
    setManagerError(null)
    setManagerNotice(null)
    try {
      const response = await apiPost<ProviderBindingStartResponse>('/provider-bindings/notion/start')
      setBinding(response)
      setManagerNotice(
        response.demo_mode
          ? `Bound ${response.account_label ?? 'Notion demo workspace'}.`
          : 'Opened Notion authorization. Complete the OAuth step to continue.',
      )
    } catch (error) {
      setManagerError(error instanceof Error ? error.message : 'Could not start Notion binding.')
    } finally {
      setIsBinding(false)
    }
  }

  async function handleCompleteBinding() {
    if (!binding || !('binding_state' in binding) || !binding.binding_state) {
      return
    }
    setIsBinding(true)
    setManagerError(null)
    setManagerNotice(null)
    try {
      const response = await apiPost<ProviderBindingSummary>('/provider-bindings/notion/complete', {
        provider: 'notion',
        binding_state: binding.binding_state,
        code: authorizationCode,
      })
      setBinding(response)
      setAuthorizationCode('')
      setManagerNotice(`Bound ${response.account_label ?? 'Notion workspace'}.`)
    } catch (error) {
      setManagerError(error instanceof Error ? error.message : 'Could not complete Notion binding.')
    } finally {
      setIsBinding(false)
    }
  }

  async function handlePreview(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setIsPreviewing(true)
    setManagerError(null)
    setManagerNotice(null)
    setPreview(null)
    try {
      const response = await apiPost<SourceConnectionPreviewResponse>('/source-connections/preview', {
        template_key: templateKey,
        provider_credential_ref: boundCredentialRef,
        source_label: sourceLabel,
        selected_scope_input: selectedScopeInput,
        visibility_class: visibilityClass,
      })
      setPreview(response)
      if (!syncIntervalSeconds) {
        setSyncIntervalSeconds(String(response.suggested_sync_interval_seconds))
      }
      setManagerNotice(`Preview resolved ${response.preview_items.length} sample items.`)
    } catch (error) {
      setManagerError(error instanceof Error ? error.message : 'Preview failed.')
    } finally {
      setIsPreviewing(false)
    }
  }

  async function handleCreate() {
    const payload: SourceConnectionCreatePayload = {
      template_key: templateKey,
      provider_credential_ref: boundCredentialRef,
      source_label: sourceLabel,
      selected_scope_input: selectedScopeInput,
      visibility_class: visibilityClass,
      sync_interval_seconds: Number(syncIntervalSeconds || recommendedInterval),
    }
    setIsCreating(true)
    setManagerError(null)
    setManagerNotice(null)
    try {
      await apiPost<SourceConnectionDetail>('/source-connections', payload)
      setPreview(null)
      setSourceLabel('')
      setSelectedScopeInput('')
      setSyncIntervalSeconds(String(recommendedInterval))
      setManagerNotice('Created source connection and started the initial sync.')
      setRefreshIndex((value) => value + 1)
    } catch (error) {
      setManagerError(error instanceof Error ? error.message : 'Create failed.')
    } finally {
      setIsCreating(false)
    }
  }

  async function handleSourceAction(
    source: SourceConnectionStatus,
    action: 'sync' | 'pause' | 'resume' | 'remove',
  ) {
    setBusyAction(`${source.id}:${action}`)
    setManagerError(null)
    setManagerNotice(null)
    try {
      if (action === 'sync') {
        await apiPost(`/source-connections/${source.id}/sync`)
      } else if (action === 'pause') {
        await apiPost(`/source-connections/${source.id}/pause`)
      } else if (action === 'resume') {
        await apiPost(`/source-connections/${source.id}/resume`)
      } else {
        await apiDelete(`/source-connections/${source.id}`)
      }
      if (editing?.connectionId === source.id && action === 'remove') {
        setEditing(null)
      }
      setManagerNotice(`${source.source_label} ${action} completed.`)
      setRefreshIndex((value) => value + 1)
    } catch (error) {
      setManagerError(error instanceof Error ? error.message : `Could not ${action} source.`)
    } finally {
      setBusyAction(null)
    }
  }

  async function handleEditStart(source: SourceConnectionStatus) {
    setBusyAction(`${source.id}:edit`)
    setManagerError(null)
    setManagerNotice(null)
    try {
      const detail = await apiGet<SourceConnectionDetail>(`/source-connections/${source.id}`)
      setEditing({
        connectionId: detail.id,
        sourceLabel: detail.source_label,
        selectedScopeInput: String(detail.selected_scope_json.input ?? detail.source_boundary_locator),
        visibilityClass: detail.visibility_class as 'member_visible' | 'evidence_only',
        syncIntervalSeconds: detail.sync_interval_seconds,
        providerCredentialRef: detail.provider_credential_ref ?? boundCredentialRef ?? '',
      })
    } catch (error) {
      setManagerError(error instanceof Error ? error.message : 'Could not load connection detail.')
    } finally {
      setBusyAction(null)
    }
  }

  async function handleEditSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!editing) {
      return
    }
    setBusyAction(`${editing.connectionId}:update`)
    setManagerError(null)
    setManagerNotice(null)
    const payload: SourceConnectionUpdatePayload = {
      source_label: editing.sourceLabel,
      selected_scope_input: editing.selectedScopeInput,
      visibility_class: editing.visibilityClass,
      sync_interval_seconds: editing.syncIntervalSeconds,
      provider_credential_ref: editing.providerCredentialRef || undefined,
    }
    try {
      await apiPatch<SourceConnectionDetail>(`/source-connections/${editing.connectionId}`, payload)
      setEditing(null)
      setManagerNotice('Updated source connection and scheduled a recovery sync.')
      setRefreshIndex((value) => value + 1)
    } catch (error) {
      setManagerError(error instanceof Error ? error.message : 'Could not update source.')
    } finally {
      setBusyAction(null)
    }
  }

  async function handleToggleRuns(source: SourceConnectionStatus) {
    if (expandedRunsId === source.id) {
      setExpandedRunsId(null)
      return
    }
    setBusyAction(`${source.id}:runs`)
    setManagerError(null)
    try {
      const runs = await apiGet<SyncRunSummary[]>(`/source-connections/${source.id}/runs`)
      setRunsByConnection((current) => ({ ...current, [source.id]: runs }))
      setExpandedRunsId(source.id)
    } catch (error) {
      setManagerError(error instanceof Error ? error.message : 'Could not load sync runs.')
    } finally {
      setBusyAction(null)
    }
  }

  return (
    <section className="page-stack sources-page">
      <PageHeader
        title="Source status"
        description="Shared connector state, source freshness, and manager controls for Notion-backed workspace memory."
      />

      {canManage ? (
        <article className="panel connector-manager-panel">
          <div className="page-header compact-header">
            <div>
              <span className="eyebrow">Connector manager</span>
              <h3>Bind Notion and create a source connection</h3>
              <p>Choose a template, authorize a workspace binding, preview the scope, then create the connection.</p>
            </div>
          </div>

          {templates.error ? (
            <div className="alert-row error-text" role="alert">
              {templates.error}
            </div>
          ) : null}
          {managerError ? (
            <div className="alert-row error-text" role="alert">
              {managerError}
            </div>
          ) : null}
          {managerNotice ? <div className="alert-row info-text">{managerNotice}</div> : null}

          <div className="manager-grid">
            <form className="source-manager-form" onSubmit={handlePreview}>
              <label className="form-field">
                <span className="mini-label">Template</span>
                <select
                  value={templateKey}
                  onChange={(event) => {
                    const nextTemplate = availableTemplates.find(
                      (template) => template.template_key === event.target.value,
                    )
                    setTemplateKey(event.target.value)
                    setVisibilityClass(nextTemplate?.default_visibility_class ?? 'member_visible')
                    setSyncIntervalSeconds(String(nextTemplate?.recommended_sync_interval_seconds ?? 900))
                    setPreview(null)
                  }}
                >
                  {availableTemplates.map((template) => (
                    <option key={template.template_key} value={template.template_key}>
                      {template.label}
                    </option>
                  ))}
                </select>
              </label>

              <div className="binding-row">
                <div>
                  <span className="mini-label">Provider binding</span>
                  <p>{boundCredentialRef ? `Ready · ${boundCredentialRef}` : 'No Notion workspace bound yet.'}</p>
                </div>
                <button type="button" className="ghost-button" onClick={handleStartBinding} disabled={isBinding}>
                  {isBinding ? 'Starting…' : 'Bind Notion'}
                </button>
              </div>

              {binding && 'authorization_url' in binding && binding.authorization_url && !boundCredentialRef ? (
                <div className="oauth-complete-form">
                  <a href={binding.authorization_url} target="_blank" rel="noreferrer">
                    Open Notion authorization
                  </a>
                  <label className="form-field">
                    <span className="mini-label">Authorization code</span>
                    <input
                      value={authorizationCode}
                      onChange={(event) => setAuthorizationCode(event.target.value)}
                      placeholder="Paste the code returned by Notion"
                    />
                  </label>
                  <button
                    type="button"
                    disabled={!authorizationCode || isBinding}
                    onClick={handleCompleteBinding}
                  >
                    {isBinding ? 'Completing…' : 'Complete binding'}
                  </button>
                </div>
              ) : null}

              <label className="form-field">
                <span className="mini-label">Source label</span>
                <input
                  value={sourceLabel}
                  onChange={(event) => setSourceLabel(event.target.value)}
                  placeholder="Engineering handbook"
                />
              </label>

              <label className="form-field">
                <span className="mini-label">
                  {selectedTemplate?.scope_kind === 'database' ? 'Database URL or UUID' : 'Page URL or UUID'}
                </span>
                <input
                  value={selectedScopeInput}
                  onChange={(event) => setSelectedScopeInput(event.target.value)}
                  placeholder={
                    selectedTemplate?.scope_kind === 'database'
                      ? 'https://www.notion.so/... or UUID'
                      : 'https://www.notion.so/... or UUID'
                  }
                />
              </label>

              <div className="compact-form-grid">
                <label className="form-field">
                  <span className="mini-label">Visibility</span>
                  <select
                    value={visibilityClass}
                    onChange={(event) =>
                      setVisibilityClass(event.target.value as 'member_visible' | 'evidence_only')
                    }
                  >
                    <option value="member_visible">member_visible</option>
                    <option value="evidence_only">evidence_only</option>
                  </select>
                </label>
                <label className="form-field">
                  <span className="mini-label">Sync interval (seconds)</span>
                  <input
                    value={syncIntervalSeconds}
                    onChange={(event) => setSyncIntervalSeconds(event.target.value)}
                    inputMode="numeric"
                    placeholder={String(recommendedInterval)}
                  />
                </label>
              </div>

              <div className="source-actions">
                <button
                  type="submit"
                  className="ghost-button"
                  disabled={!sourceLabel || !selectedScopeInput || !boundCredentialRef || isPreviewing}
                >
                  {isPreviewing ? 'Previewing…' : 'Preview'}
                </button>
                <button
                  type="button"
                  disabled={!preview || isCreating}
                  onClick={handleCreate}
                >
                  {isCreating ? 'Creating…' : 'Create connection'}
                </button>
              </div>
            </form>

            <div className="panel nested-panel preview-panel">
              <div className="panel-heading panel-heading-start">
                <div>
                  <span className="eyebrow">Preview</span>
                  <h3>{preview ? preview.resolved_source_boundary_locator : 'No preview yet'}</h3>
                </div>
                {preview ? <StatusPill value={preview.visibility_class} /> : null}
              </div>
              {preview ? (
                <>
                  <p className="muted">
                    {preview.suggested_sync_mode} every {preview.suggested_sync_interval_seconds} seconds
                  </p>
                  <ul className="stack-list preview-list">
                    {preview.preview_items.map((item) => (
                      <li key={item.upstream_id} className="list-card compact-card">
                        <strong>{item.title}</strong>
                        <p>{item.excerpt ?? 'No preview excerpt available.'}</p>
                        <div className="inline-meta">
                          <StatusPill value={item.artifact_type} />
                          {item.source_updated_at ? (
                            <time className="muted" dateTime={item.source_updated_at}>
                              {formatLocalDateTime(item.source_updated_at)}
                            </time>
                          ) : null}
                        </div>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <EmptyState
                  title="Preview required"
                  description="Resolve the scope before creating the connection so the operator can inspect sample source memory."
                />
              )}
            </div>
          </div>
        </article>
      ) : null}

      <div className="page-stack">
        {(sources.data ?? []).map((source) => (
          <article key={source.id} className="panel nested-panel source-card">
            <div className="page-header compact-header source-card-header">
              <div>
                <span className="eyebrow">{source.provider}</span>
                <h3>{source.source_label}</h3>
                <p className="code-text source-locator">{source.source_boundary_locator}</p>
              </div>
              <div className="inline-meta">
                <StatusPill value={source.source_connection_state} />
                <StatusPill value={source.freshness_state} />
              </div>
            </div>

            <div className="meta-grid compact-columns source-meta-grid">
              <div>
                <span className="mini-label">Template</span>
                <p>{source.template_key}</p>
              </div>
              <div>
                <span className="mini-label">Visibility</span>
                <p>{source.visibility_class}</p>
              </div>
              <div>
                <span className="mini-label">Last success</span>
                <p>{renderDateTime(source.last_successful_sync_at)}</p>
              </div>
              <div>
                <span className="mini-label">Last run</span>
                <p>{renderDateTime(source.last_run_at)}</p>
              </div>
              <div>
                <span className="mini-label">Run status</span>
                <p>{source.last_run_status ?? 'unknown'}</p>
              </div>
              <div>
                <span className="mini-label">Next scheduled</span>
                <p>{renderDateTime(source.next_scheduled_sync_at, 'Not scheduled')}</p>
              </div>
            </div>

            {source.last_error ? (
              <div className="alert-row error-text" role="alert">
                {source.last_error}
              </div>
            ) : (
              <div className="alert-row info-text">{recoverabilityHint(source.source_connection_state)}</div>
            )}

            {source.can_manage ? (
              <div className="source-actions">
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => handleSourceAction(source, 'sync')}
                  disabled={busyAction === `${source.id}:sync`}
                >
                  {busyAction === `${source.id}:sync` ? 'Syncing…' : 'Resync'}
                </button>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() =>
                    handleSourceAction(
                      source,
                      source.source_connection_state === 'paused' ? 'resume' : 'pause',
                    )
                  }
                  disabled={
                    busyAction === `${source.id}:pause` || busyAction === `${source.id}:resume`
                  }
                >
                  {source.source_connection_state === 'paused'
                    ? busyAction === `${source.id}:resume`
                      ? 'Resuming…'
                      : 'Resume'
                    : busyAction === `${source.id}:pause`
                      ? 'Pausing…'
                      : 'Pause'}
                </button>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => handleEditStart(source)}
                  disabled={busyAction === `${source.id}:edit`}
                >
                  {busyAction === `${source.id}:edit` ? 'Loading…' : 'Edit'}
                </button>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => handleToggleRuns(source)}
                  disabled={busyAction === `${source.id}:runs`}
                >
                  {expandedRunsId === source.id ? 'Hide runs' : busyAction === `${source.id}:runs` ? 'Loading…' : 'Show runs'}
                </button>
                <button
                  type="button"
                  className="destructive-button"
                  onClick={() => handleSourceAction(source, 'remove')}
                  disabled={busyAction === `${source.id}:remove`}
                >
                  {busyAction === `${source.id}:remove` ? 'Removing…' : 'Remove'}
                </button>
              </div>
            ) : null}

            {editing?.connectionId === source.id ? (
              <form className="panel nested-panel source-editor" onSubmit={handleEditSubmit}>
                <div className="compact-form-grid">
                  <label className="form-field">
                    <span className="mini-label">Source label</span>
                    <input
                      value={editing.sourceLabel}
                      onChange={(event) =>
                        setEditing((current) =>
                          current ? { ...current, sourceLabel: event.target.value } : current,
                        )
                      }
                    />
                  </label>
                  <label className="form-field">
                    <span className="mini-label">Scope input</span>
                    <input
                      value={editing.selectedScopeInput}
                      onChange={(event) =>
                        setEditing((current) =>
                          current ? { ...current, selectedScopeInput: event.target.value } : current,
                        )
                      }
                    />
                  </label>
                  <label className="form-field">
                    <span className="mini-label">Visibility</span>
                    <select
                      value={editing.visibilityClass}
                      onChange={(event) =>
                        setEditing((current) =>
                          current
                            ? {
                                ...current,
                                visibilityClass: event.target.value as 'member_visible' | 'evidence_only',
                              }
                            : current,
                        )
                      }
                    >
                      <option value="member_visible">member_visible</option>
                      <option value="evidence_only">evidence_only</option>
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="mini-label">Sync interval (seconds)</span>
                    <input
                      value={String(editing.syncIntervalSeconds)}
                      onChange={(event) =>
                        setEditing((current) =>
                          current
                            ? {
                                ...current,
                                syncIntervalSeconds: Number(event.target.value || current.syncIntervalSeconds),
                              }
                            : current,
                        )
                      }
                      inputMode="numeric"
                    />
                  </label>
                </div>
                <div className="source-actions">
                  <button type="submit" disabled={busyAction === `${source.id}:update`}>
                    {busyAction === `${source.id}:update` ? 'Saving…' : 'Save changes'}
                  </button>
                  <button type="button" className="ghost-button" onClick={() => setEditing(null)}>
                    Cancel
                  </button>
                </div>
              </form>
            ) : null}

            {expandedRunsId === source.id ? (
              <div className="panel nested-panel source-runs">
                <div className="panel-heading panel-heading-start">
                  <div>
                    <span className="eyebrow">Recent sync runs</span>
                    <h3>{source.source_label}</h3>
                  </div>
                </div>
                <ul className="stack-list runs-list">
                  {(runsByConnection[source.id] ?? []).map((run) => (
                    <li key={run.id} className="list-card compact-card">
                      <div className="inline-meta">
                        <StatusPill value={run.run_status} />
                        <StatusPill value={run.trigger_kind} />
                      </div>
                      <p>
                        Started {formatLocalDateTime(run.started_at)} · Artifacts {run.artifact_count} · Support items{' '}
                        {run.support_item_count}
                      </p>
                      {run.error_summary ? <p className="error-text">{run.error_summary}</p> : null}
                    </li>
                  ))}
                </ul>
                {(runsByConnection[source.id] ?? []).length === 0 ? (
                  <p className="muted">No sync runs have been recorded yet.</p>
                ) : null}
              </div>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  )
}

function recoverabilityHint(state: string) {
  if (state === 'degraded') {
    return 'Check the provider binding or selected scope, then run a recovery sync.'
  }
  if (state === 'paused') {
    return 'Resume this connection when workspace memory should refresh again.'
  }
  if (state === 'removed') {
    return 'Removed sources stay visible for provenance but no longer schedule sync work.'
  }
  return 'Healthy sources refresh on schedule and expose lineage through synced artifacts and evidence.'
}
