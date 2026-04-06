type StatusPillProps = {
  value: string
}

export function StatusPill({ value }: StatusPillProps) {
  const normalized = value.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
  return <span className={`status-pill ${normalized}`}>{value.replaceAll('_', ' ')}</span>
}
