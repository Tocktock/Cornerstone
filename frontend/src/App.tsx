import { BrowserRouter, Route, Routes } from 'react-router-dom'

import { apiGet } from './api/client'
import { EmptyState } from './components/EmptyState'
import { Layout } from './components/Layout'
import { useAsyncData } from './hooks/useAsyncData'
import { ArtifactsPage } from './pages/ArtifactsPage'
import { DashboardPage } from './pages/DashboardPage'
import { DecisionsPage } from './pages/DecisionsPage'
import { GlossaryPage } from './pages/GlossaryPage'
import { GraphPage } from './pages/GraphPage'
import { ReviewPage } from './pages/ReviewPage'
import { SourcesPage } from './pages/SourcesPage'
import type { Actor, ContextSpace } from './types/api'

export function App() {
  const contextSpaces = useAsyncData<ContextSpace[]>(() => apiGet('/context-spaces'), [])
  const contextSpace = contextSpaces.data?.[0] ?? null
  const actors = useAsyncData<Actor[]>(
    () => apiGet('/actors', { context_space_id: contextSpace?.id }),
    [contextSpace?.id],
  )
  const reviewerId =
    actors.data?.find((actor) => actor.roles.some((role) => role.toLowerCase() === 'reviewer'))?.id ?? null

  if (contextSpaces.isLoading) {
    return <div className="loading-screen">Loading Cornerstone workspace…</div>
  }

  if (contextSpaces.error) {
    return <EmptyState title="Frontend failed to load" description={contextSpaces.error} />
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout contextSpace={contextSpace} />}>
          <Route index element={<DashboardPage contextSpaceId={contextSpace?.id ?? null} />} />
          <Route path="glossary" element={<GlossaryPage contextSpaceId={contextSpace?.id ?? null} />} />
          <Route path="graph" element={<GraphPage contextSpaceId={contextSpace?.id ?? null} />} />
          <Route path="decisions" element={<DecisionsPage contextSpaceId={contextSpace?.id ?? null} />} />
          <Route path="artifacts" element={<ArtifactsPage contextSpaceId={contextSpace?.id ?? null} />} />
          <Route
            path="review"
            element={<ReviewPage contextSpaceId={contextSpace?.id ?? null} reviewerId={reviewerId} />}
          />
          <Route path="sources" element={<SourcesPage contextSpaceId={contextSpace?.id ?? null} />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
