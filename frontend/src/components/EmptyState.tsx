import type { ReactNode } from 'react'

type EmptyStateProps = {
  title: string
  description: string
  eyebrow?: string
  actions?: ReactNode
}

export function EmptyState({
  title,
  description,
  eyebrow = 'Cornerstone state',
  actions,
}: EmptyStateProps) {
  return (
    <div className="empty-state panel">
      <span className="eyebrow">{eyebrow}</span>
      <h3>{title}</h3>
      <p>{description}</p>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </div>
  )
}
