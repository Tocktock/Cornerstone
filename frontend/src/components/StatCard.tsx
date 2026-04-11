type StatCardProps = {
  label: string
  value: number
  helper: string
  tone?: 'default' | 'info' | 'danger' | 'sage'
}

export function StatCard({
  label,
  value,
  helper,
  tone = 'default',
}: StatCardProps) {
  return (
    <article className={`stat-card panel stat-card-${tone}`}>
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      <p>{helper}</p>
    </article>
  )
}
