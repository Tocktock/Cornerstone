type StatCardProps = {
  label: string
  value: number
  helper: string
}

export function StatCard({ label, value, helper }: StatCardProps) {
  return (
    <article className="stat-card panel">
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      <p>{helper}</p>
    </article>
  )
}
