import { useEffect, useMemo, useState } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'

import { apiGet, getActorToken, setActorToken } from './api/client'
import { EmptyState } from './components/EmptyState'
import { Layout } from './components/Layout'
import { useAsyncData } from './hooks/useAsyncData'
import { DashboardPage } from './pages/DashboardPage'
import { DecisionsPage } from './pages/DecisionsPage'
import { GlossaryPage } from './pages/GlossaryPage'
import { GraphPage } from './pages/GraphPage'
import { ReviewPage } from './pages/ReviewPage'
import { SourcesPage } from './pages/SourcesPage'
import type { ViewerBootstrap } from './types/api'

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
              actors={actors}
              activeActor={activeActor}
              onActorChange={handleActorChange}
            />
          }
        >
          <Route index element={<DashboardPage workspace={bootstrap.data.workspace} activeActor={activeActor} />} />
          <Route path="glossary" element={<GlossaryPage workspace={bootstrap.data.workspace} />} />
          <Route path="graph" element={<GraphPage workspace={bootstrap.data.workspace} />} />
          <Route path="decisions" element={<DecisionsPage workspace={bootstrap.data.workspace} />} />
          <Route path="review" element={<ReviewPage workspace={bootstrap.data.workspace} activeActor={activeActor} />} />
          <Route path="sources" element={<SourcesPage workspace={bootstrap.data.workspace} />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
