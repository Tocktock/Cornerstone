type StatusPillProps = {
  value: string
}

export function StatusPill({ value }: StatusPillProps) {
  const normalized = value.toLowerCase()
  return <span className={`status-pill ${normalized}`}>{value.replaceAll('_', ' ')}</span>
}
