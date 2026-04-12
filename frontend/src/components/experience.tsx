import type { ReactNode } from 'react'
import { Link, NavLink } from 'react-router-dom'

import { StatusPill } from './StatusPill'
import type { ProvenanceSummary, ResourceRef } from '../types/api'

type ArtifactCardProps = {
  to: string
  eyebrow: string
  title: string
  summary: string
  supportVisibility: string
  lifecycleState: string
  verificationState?: string | null
  variant?: 'lead' | 'standard' | 'compact' | 'rail'
  ctaLabel?: string
  meta?: ReactNode
  children?: ReactNode
}

type ProvenanceStripProps = {
  summary: ProvenanceSummary
  supportVisibility: string
  verificationState?: string | null
  variant?: 'standard' | 'rail' | 'compact'
}

type AlertBannerProps = {
  tone?: 'default' | 'danger'
  eyebrow?: string
  title: string
  description: string
}

type LineageRailProps = {
  previous?: { label: string; to?: string | null } | null
  next?: { label: string; to?: string | null } | null
  variant?: 'standard' | 'timeline'
}

type ExploreTabsProps = {
  basePath?: string
}

type SectionIntroProps = {
  eyebrow: string
  title: string
  description?: string
  actions?: ReactNode
  compact?: boolean
}

export function ExploreTabs({ basePath = '/explore' }: ExploreTabsProps) {
  const items = [
    { to: `${basePath}/topics`, label: 'Topics' },
    { to: `${basePath}/decisions`, label: 'Decisions' },
    { to: `${basePath}/map`, label: 'Map' },
  ]

  return (
    <div className="explore-tabs-wrap">
      <span className="mini-label">Explore views</span>
      <nav className="explore-tabs" aria-label="Explore views">
        {items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `explore-tab ${isActive ? 'active' : ''}`}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </div>
  )
}

export function ArtifactCard({
  to,
  eyebrow,
  title,
  summary,
  supportVisibility,
  lifecycleState,
  verificationState,
  variant = 'standard',
  ctaLabel = 'Open',
  meta,
  children,
}: ArtifactCardProps) {
  return (
    <article className={`artifact-card artifact-card-${variant}`}>
      <div className="artifact-card-header">
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h3>{title}</h3>
        </div>
        <Link className="artifact-link" to={to}>
          {ctaLabel}
        </Link>
      </div>
      <div className="artifact-status-row">
        <StatusPill value={lifecycleState} />
        <StatusPill value={supportVisibility} />
        {verificationState ? <StatusPill value={verificationState} /> : null}
      </div>
      <p className="artifact-card-summary">{summary}</p>
      {meta ? <div className="artifact-meta">{meta}</div> : null}
      {children}
    </article>
  )
}

export function ProvenanceStrip({
  summary,
  supportVisibility,
  verificationState,
  variant = 'standard',
}: ProvenanceStripProps) {
  const resolvedVerificationState = verificationState ?? summary.verification_state ?? null

  return (
    <section className={`provenance-strip provenance-strip-${variant}`} aria-label="Trust and provenance">
      <div className="artifact-status-row">
        <StatusPill value={supportVisibility} />
        {resolvedVerificationState ? <StatusPill value={resolvedVerificationState} /> : null}
        <StatusPill value={summary.freshness_state} />
      </div>
      <div className="provenance-stats">
        <span>{summary.visible_support_item_count} visible support</span>
        <span>{summary.support_item_count} total support</span>
        <span>{summary.restricted_support_present ? 'restricted support present' : 'no restricted support'}</span>
        <span>{summary.promotion_lineage_present ? 'promotion lineage present' : 'no promotion lineage'}</span>
      </div>
    </section>
  )
}

export function LineageRail({ previous, next, variant = 'standard' }: LineageRailProps) {
  if (!previous && !next) {
    return null
  }

  return (
    <section className={`lineage-rail lineage-rail-${variant}`} aria-label="Decision lineage">
      {previous ? (
        <div className="lineage-stop">
          <span className="mini-label">Supersedes</span>
          {previous.to ? <Link to={previous.to}>{previous.label}</Link> : <span>{previous.label}</span>}
        </div>
      ) : (
        <div className="lineage-stop">
          <span className="mini-label">Supersedes</span>
          <span>None</span>
        </div>
      )}
      {next ? (
        <div className="lineage-stop">
          <span className="mini-label">Superseded by</span>
          {next.to ? <Link to={next.to}>{next.label}</Link> : <span>{next.label}</span>}
        </div>
      ) : (
        <div className="lineage-stop">
          <span className="mini-label">Superseded by</span>
          <span>Current endpoint</span>
        </div>
      )}
    </section>
  )
}

export function AlertBanner({
  tone = 'default',
  eyebrow,
  title,
  description,
}: AlertBannerProps) {
  return (
    <div className={`alert-banner ${tone}`}>
      {eyebrow ? <span className="eyebrow">{eyebrow}</span> : null}
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  )
}

export function SectionIntro({
  eyebrow,
  title,
  description,
  actions,
  compact = false,
}: SectionIntroProps) {
  return (
    <div className={`section-intro ${compact ? 'compact' : ''}`}>
      <div className="section-intro-copy">
        <span className="eyebrow">{eyebrow}</span>
        <h3>{title}</h3>
        {description ? <p className="panel-copy">{description}</p> : null}
      </div>
      {actions ? <div className="section-intro-actions">{actions}</div> : null}
    </div>
  )
}

export function RefList({
  title,
  refs,
  resolveHref,
}: {
  title: string
  refs: ResourceRef[]
  resolveHref?: (ref: ResourceRef) => string | null
}) {
  if (!refs.length) {
    return null
  }

  return (
    <section className="narrative-section">
      <h4>{title}</h4>
      <div className="chip-row">
        {refs.map((ref) => {
          const href = resolveHref?.(ref) ?? null
          return href ? (
            <Link key={ref.resource_id} className="chip" to={href}>
              {ref.resource_label}
            </Link>
          ) : (
            <span key={ref.resource_id} className="chip">
              {ref.resource_label}
            </span>
          )
        })}
      </div>
    </section>
  )
}
