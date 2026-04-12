import { useEffect, useMemo, useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { apiGet, getActorToken, setActorToken } from './api/client'
import { EmptyState } from './components/EmptyState'
import { Layout } from './components/Layout'
import { useAsyncData } from './hooks/useAsyncData'
import { ConceptDetailPage } from './pages/ConceptDetailPage'
import { DecisionDetailPage } from './pages/DecisionDetailPage'
import { ExploreDecisionsPage } from './pages/ExploreDecisionsPage'
import { ExploreMapPage } from './pages/ExploreMapPage'
import { ExploreTopicsPage } from './pages/ExploreTopicsPage'
import { ReviewStudioPage } from './pages/ReviewStudioPage'
import { SourceStudioPage } from './pages/SourceStudioPage'
import { WorkspacePage } from './pages/WorkspacePage'
import type { ViewerBootstrap } from './types/api'
import { canActorManageConnectors, canActorReview } from './viewModels'

export function App() {
  const bootstrap = useAsyncData<ViewerBootstrap>(() => apiGet('/bootstrap'), [])
  const [selectedActorId, setSelectedActorId] = useState<string | null>(null)
  const [sessionToken, setSessionToken] = useState<string | null>(() => getActorToken())

  const actors = bootstrap.data?.actors ?? []
  const activeActor = useMemo(() => {
    if (!actors.length) {
      return null
    }
    return (
      actors.find((actor) => actor.actor_id === selectedActorId) ??
      actors.find((actor) => actor.token === getActorToken()) ??
      actors.find((actor) => actor.preferred_consumer_scope === 'member') ??
      actors[0]
    )
  }, [actors, selectedActorId])

  useEffect(() => {
    if (!activeActor) {
      return
    }
    if (selectedActorId !== activeActor.actor_id) {
      setSelectedActorId(activeActor.actor_id)
    }
    if (sessionToken !== activeActor.token) {
      setActorToken(activeActor.token)
      setSessionToken(activeActor.token)
    }
  }, [activeActor, selectedActorId, sessionToken])

  if (bootstrap.isLoading) {
    return <div className="loading-screen">Bootstrapping Cornerstone…</div>
  }

  if (bootstrap.error || !bootstrap.data || !activeActor) {
    return (
      <EmptyState
        title="Frontend failed to load"
        description={bootstrap.error ?? 'Bootstrap payload was not available.'}
      />
    )
  }

  if (sessionToken !== activeActor.token) {
    return <div className="loading-screen">Preparing actor session…</div>
  }

  const reviewAccess = canActorReview(activeActor)
  const connectorAccess = canActorManageConnectors(activeActor)

  function handleActorChange(actorId: string) {
    const nextActor = actors.find((actor) => actor.actor_id === actorId)
    if (nextActor) {
      setActorToken(nextActor.token)
      setSessionToken(nextActor.token)
    }
    setSelectedActorId(actorId)
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <Layout
              workspace={bootstrap.data.workspace}
              runtimeInfo={bootstrap.data}
              actors={actors}
              activeActor={activeActor}
              canReview={reviewAccess}
              canManageConnectors={connectorAccess}
              onActorChange={handleActorChange}
            />
          }
        >
          <Route index element={<WorkspacePage activeActor={activeActor} runtimeInfo={bootstrap.data} />} />
          <Route
            path="explore/topics"
            element={<ExploreTopicsPage activeActor={activeActor} runtimeInfo={bootstrap.data} />}
          />
          <Route
            path="explore/decisions"
            element={<ExploreDecisionsPage activeActor={activeActor} runtimeInfo={bootstrap.data} />}
          />
          <Route
            path="explore/map"
            element={<ExploreMapPage activeActor={activeActor} runtimeInfo={bootstrap.data} />}
          />
          <Route
            path="explore/map/:conceptId"
            element={<ExploreMapPage activeActor={activeActor} runtimeInfo={bootstrap.data} />}
          />
          <Route path="concepts/:publicSlug" element={<ConceptDetailPage activeActor={activeActor} />} />
          <Route path="decisions/:publicSlug" element={<DecisionDetailPage activeActor={activeActor} />} />
          <Route
            path="review-studio"
            element={<ReviewStudioPage activeActor={activeActor} actors={actors} canReview={reviewAccess} />}
          />
          <Route
            path="source-studio"
            element={<SourceStudioPage activeActor={activeActor} runtimeInfo={bootstrap.data} />}
          />

          <Route path="glossary" element={<Navigate to="/explore/topics" replace />} />
          <Route path="graph" element={<Navigate to="/explore/map" replace />} />
          <Route path="decisions" element={<Navigate to="/explore/decisions" replace />} />
          <Route path="review" element={<Navigate to="/review-studio" replace />} />
          <Route path="sources" element={<Navigate to="/source-studio" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
