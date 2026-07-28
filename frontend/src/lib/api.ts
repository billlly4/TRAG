import { supabase } from './supabase'
import type { ChatFrame, DocumentMeta, Message, Thread } from './types'

const BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function authHeaders(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  if (!token) throw new Error('Not signed in')
  return { Authorization: `Bearer ${token}` }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { ...(await authHeaders()), ...(init.headers ?? {}) },
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${body.slice(0, 300)}`)
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T)
}

export const listThreads = () => request<Thread[]>('/api/threads')

export const createThread = () =>
  request<Thread>('/api/threads', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  })

export const deleteThread = (id: string) =>
  request<void>(`/api/threads/${id}`, { method: 'DELETE' })

export const listMessages = (id: string) =>
  request<Message[]>(`/api/threads/${id}/messages`)

export const listDocuments = () => request<DocumentMeta[]>('/api/files')

export const deleteDocument = (id: string) =>
  request<void>(`/api/files/${id}`, { method: 'DELETE' })

export async function uploadDocument(file: File): Promise<DocumentMeta> {
  const form = new FormData()
  form.append('file', file)
  // No Content-Type header: the browser must set the multipart boundary.
  return request<DocumentMeta>('/api/files', { method: 'POST', body: form })
}

interface StreamHandlers {
  onDelta: (text: string) => void
  onToolUse: (frame: Extract<ChatFrame, { type: 'tool_use' }>) => void
  onToolResult: (frame: Extract<ChatFrame, { type: 'tool_result' }>) => void
  onDone: (frame: Extract<ChatFrame, { type: 'done' }>) => void
  onError: (detail: string) => void
}

/**
 * Consume the SSE stream from POST /api/chat.
 *
 * EventSource is unusable here: it cannot set an Authorization header and only
 * issues GET requests. So we read the response body ourselves and parse SSE
 * frames by hand.
 */
export async function streamChat(
  threadId: string,
  message: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response
  try {
    res = await fetch(`${BASE}/api/chat`, {
      method: 'POST',
      headers: { ...(await authHeaders()), 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: threadId, message }),
      signal,
    })
  } catch (err) {
    handlers.onError(err instanceof Error ? err.message : String(err))
    return
  }

  if (!res.ok || !res.body) {
    handlers.onError(`${res.status} ${(await res.text()).slice(0, 300)}`)
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    // stream: true keeps multi-byte characters intact across chunk boundaries.
    buffer += decoder.decode(value, { stream: true })

    // Frames are separated by a blank line. The trailing element is whatever
    // arrived mid-frame, so it stays in the buffer for the next read.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const raw of frames) {
      const line = raw.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue

      let frame: ChatFrame
      try {
        frame = JSON.parse(line.slice(6)) as ChatFrame
      } catch {
        continue
      }

      if (frame.type === 'delta') handlers.onDelta(frame.text)
      else if (frame.type === 'tool_use') handlers.onToolUse(frame)
      else if (frame.type === 'tool_result') handlers.onToolResult(frame)
      else if (frame.type === 'done') handlers.onDone(frame)
      else if (frame.type === 'error') handlers.onError(frame.detail)
    }
  }
}
