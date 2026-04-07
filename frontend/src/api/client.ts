const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1'
const actorTokenStorageKey = 'cornerstone.actorToken'

let actorToken = typeof window !== 'undefined' ? window.localStorage.getItem(actorTokenStorageKey) : null

export function setActorToken(token: string | null) {
  actorToken = token
  if (typeof window !== 'undefined') {
    if (token) {
      window.localStorage.setItem(actorTokenStorageKey, token)
    } else {
      window.localStorage.removeItem(actorTokenStorageKey)
    }
  }
}

export function getActorToken() {
  return actorToken
}

async function apiFetch<T>(path: string, init?: RequestInit, params?: Record<string, string | undefined>): Promise<T> {
  const url = new URL(`${apiBaseUrl}${path}`)
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value) {
        url.searchParams.set(key, value)
      }
    }
  }

  const headers = new Headers(init?.headers ?? {})
  if (actorToken) {
    headers.set('Authorization', `Bearer ${actorToken}`)
  }
  if (!headers.has('Content-Type') && init?.body) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(url.toString(), {
    ...init,
    headers,
  })
  if (!response.ok) {
    const text = await response.text()
    try {
      const parsed = JSON.parse(text) as { detail?: string }
      throw new Error(parsed.detail ?? text)
    } catch {
      throw new Error(text || `Request failed with status ${response.status}`)
    }
  }
  return response.json() as Promise<T>
}

export function apiGet<T>(path: string, params?: Record<string, string | undefined>) {
  return apiFetch<T>(path, undefined, params)
}

export function apiPost<T>(path: string, body?: unknown, params?: Record<string, string | undefined>) {
  return apiFetch<T>(
    path,
    {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
    },
    params,
  )
}

export function apiPatch<T>(path: string, body?: unknown, params?: Record<string, string | undefined>) {
  return apiFetch<T>(
    path,
    {
      method: 'PATCH',
      body: body ? JSON.stringify(body) : undefined,
    },
    params,
  )
}

export function apiDelete<T>(path: string, body?: unknown, params?: Record<string, string | undefined>) {
  return apiFetch<T>(
    path,
    {
      method: 'DELETE',
      body: body ? JSON.stringify(body) : undefined,
    },
    params,
  )
}
