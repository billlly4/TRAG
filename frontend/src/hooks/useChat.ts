import { useCallback, useEffect, useRef, useState } from 'react'

import * as api from '../lib/api'
import { supabase } from '../lib/supabase'
import type { DocumentMeta, Message, Source, Thread } from '../lib/types'

/** A search the assistant is running (or has run) during the live turn. */
export interface LiveTool {
  query: string
  /** null while the search is still executing */
  sources: Source[] | null
  isError: boolean
}

export function useChat() {
  const [threads, setThreads] = useState<Thread[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [documents, setDocuments] = useState<DocumentMeta[]>([])

  const [streaming, setStreaming] = useState(false)
  const [streamText, setStreamText] = useState('')
  const [liveTools, setLiveTools] = useState<LiveTool[]>([])
  const [error, setError] = useState<string | null>(null)
  const [truncated, setTruncated] = useState(false)

  const abortRef = useRef<AbortController | null>(null)

  const refreshThreads = useCallback(async () => {
    setThreads(await api.listThreads())
  }, [])

  const refreshDocuments = useCallback(async () => {
    setDocuments(await api.listDocuments())
  }, [])

  useEffect(() => {
    refreshThreads().catch((e) => setError(String(e)))
    refreshDocuments().catch((e) => setError(String(e)))
  }, [refreshThreads, refreshDocuments])

  // Ingestion status arrives by Realtime push, not polling: the backend writes
  // documents.status at each pipeline transition and Postgres publishes the
  // change. RLS scopes events to this user's rows.
  useEffect(() => {
    const channel = supabase
      .channel('documents-status')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'documents' },
        (payload) => {
          if (payload.eventType === 'DELETE') {
            const old = payload.old as { id?: string }
            if (old.id) setDocuments((prev) => prev.filter((d) => d.id !== old.id))
            return
          }
          const row = payload.new as DocumentMeta
          setDocuments((prev) =>
            prev.some((d) => d.id === row.id)
              ? prev.map((d) => (d.id === row.id ? { ...d, ...row } : d))
              : [row, ...prev],
          )
        },
      )
      .subscribe()
    return () => {
      void supabase.removeChannel(channel)
    }
  }, [])

  // Loading a thread refetches from the backend rather than trusting local
  // state: the transcript lives in Postgres, and this is the path that proves
  // it survives a reload.
  useEffect(() => {
    if (!activeId) {
      setMessages([])
      return
    }
    api.listMessages(activeId).then(setMessages).catch((e) => setError(String(e)))
  }, [activeId])

  const newThread = useCallback(async () => {
    const thread = await api.createThread()
    setThreads((prev) => [thread, ...prev])
    setActiveId(thread.id)
    return thread
  }, [])

  const removeThread = useCallback(
    async (id: string) => {
      await api.deleteThread(id)
      setThreads((prev) => prev.filter((t) => t.id !== id))
      setActiveId((cur) => (cur === id ? null : cur))
    },
    [],
  )

  const send = useCallback(
    async (text: string) => {
      let threadId = activeId
      if (!threadId) threadId = (await newThread()).id

      setError(null)
      setTruncated(false)
      setStreaming(true)
      setStreamText('')
      setLiveTools([])

      // Optimistic user message so the input clears and the turn appears
      // immediately, matching what the backend has already persisted.
      setMessages((prev) => [
        ...prev,
        {
          id: `optimistic-${Date.now()}`,
          thread_id: threadId,
          role: 'user',
          content: [{ type: 'text', text }],
          created_at: new Date().toISOString(),
        },
      ])

      const controller = new AbortController()
      abortRef.current = controller

      await api.streamChat(
        threadId,
        text,
        {
          onDelta: (chunk) => setStreamText((prev) => prev + chunk),
          onToolUse: (frame) =>
            setLiveTools((prev) => [
              ...prev,
              { query: frame.input.query ?? '', sources: null, isError: false },
            ]),
          onToolResult: (frame) =>
            setLiveTools((prev) => {
              const next = [...prev]
              for (let i = next.length - 1; i >= 0; i--) {
                if (next[i].sources === null) {
                  next[i] = {
                    ...next[i],
                    sources: frame.sources,
                    isError: frame.is_error,
                  }
                  break
                }
              }
              return next
            }),
          onDone: (frame) => {
            setTruncated(frame.truncated)
            // A tool-using turn persisted several rows (assistant tool call,
            // tool results, final answer), so refetch the transcript rather
            // than appending just the last message.
            api.listMessages(threadId).then(setMessages).catch(() => {})
            setStreamText('')
            setLiveTools([])
            refreshThreads().catch(() => {})
          },
          onError: (detail) => {
            setError(detail)
            setStreamText('')
            setLiveTools([])
          },
        },
        controller.signal,
      )

      setStreaming(false)
      abortRef.current = null
    },
    [activeId, newThread, refreshThreads],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])

  const upload = useCallback(async (files: File | File[]) => {
    setError(null)
    const list = Array.isArray(files) ? files : [files]
    const uploaded: DocumentMeta[] = []

    for (const file of list) {
      try {
        const doc = await api.uploadDocument(file)
        uploaded.push(doc)
      } catch (e) {
        const detail = e instanceof Error ? e.message : String(e)
        setError(`${file.name}: ${detail}`)
      }
    }

    // Realtime will also announce these inserts; the merge in the subscription
    // handler de-duplicates by id, so adding them here is just lower latency.
    if (uploaded.length > 0)
      setDocuments((prev) => [
        ...uploaded.filter((u) => !prev.some((d) => d.id === u.id)),
        ...prev,
      ])
    return uploaded
  }, [])

  const removeDocument = useCallback(async (id: string) => {
    await api.deleteDocument(id)
    setDocuments((prev) => prev.filter((d) => d.id !== id))
  }, [])

  return {
    threads, activeId, setActiveId, messages, documents,
    streaming, streamText, liveTools, error, truncated,
    newThread, removeThread, send, stop, upload, removeDocument,
  }
}
