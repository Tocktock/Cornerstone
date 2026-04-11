type EmptyStateProps = {
  title: string
  description: string
  eyebrow?: string
}

export function EmptyState({
  title,
  description,
  eyebrow = 'Cornerstone state',
}: EmptyStateProps) {
  return (
    <div className="empty-state panel">
      <span className="eyebrow">{eyebrow}</span>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  )
}
