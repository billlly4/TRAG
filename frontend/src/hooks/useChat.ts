import { useCallback, useEffect, useRef, useState } from 'react'

import * as api from '../lib/api'
import { supabase } from '../lib/supabase'
import type { DocumentMeta, Message, Source, Thread, Usage } from '../lib/types'

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
  const [usage, setUsage] = useState<Usage | null>(null)

  const [streaming, setStreaming] = useState(false)
  const [streamText, setStreamText] = useState('')
  const [liveTools, setLiveTools] = useState<LiveTool[]>([])
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [truncated, setTruncated] = useState(false)

  const abortRef = useRef<AbortController | null>(null)
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const showNotice = useCallback((text: string) => {
    setNotice(text)
    if (noticeTimer.current) clearTimeout(noticeTimer.current)
    noticeTimer.current = setTimeout(() => setNotice(null), 6000)
  }, [])

  const refreshThreads = useCallback(async () => {
    setThreads(await api.listThreads())
  }, [])

  const refreshDocuments = useCallback(async () => {
    setDocuments(await api.listDocuments())
  }, [])

  // Refetched rather than derived locally: storage counts bytes the client
  // never sees for documents it did not upload this session, and the limits
  // themselves live in the database.
  const refreshUsage = useCallback(async () => {
    setUsage(await api.getUsage())
  }, [])

  useEffect(() => {
    refreshThreads().catch((e) => setError(String(e)))
    refreshDocuments().catch((e) => setError(String(e)))
    refreshUsage().catch(() => {})
  }, [refreshThreads, refreshDocuments, refreshUsage])

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
          // `stale` is computed by the API, not stored, so a raw table row
          // can't carry it. Terminal states refetch the list to get it right;
          // intermediate pipeline states just merge for zero-latency badges.
          if (row.status === 'ready') {
            refreshDocuments().catch(() => {})
            return
          }
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
    // Errors are surfaced, not thrown past the caller: hitting the chat quota
    // is an ordinary outcome, and an unhandled rejection here would leave the
    // button looking like it silently did nothing.
    try {
      const thread = await api.createThread()
      setThreads((prev) => [thread, ...prev])
      setActiveId(thread.id)
      setError(null)
      void refreshUsage()
      return thread
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      return null
    }
  }, [refreshUsage])

  const removeThread = useCallback(
    async (id: string) => {
      await api.deleteThread(id)
      setThreads((prev) => prev.filter((t) => t.id !== id))
      setActiveId((cur) => (cur === id ? null : cur))
      void refreshUsage()
    },
    [refreshUsage],
  )

  const send = useCallback(
    async (text: string) => {
      let threadId = activeId
      if (!threadId) {
        // newThread returns null when the chat quota is reached; the error is
        // already on screen, so stop rather than sending into a null thread.
        const created = await newThread()
        if (!created) return
        threadId = created.id
      }

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

  const upload = useCallback(
    async (files: File | File[]) => {
      setError(null)
      const list = Array.isArray(files) ? files : [files]
      const uploaded: DocumentMeta[] = []
      const unchanged: string[] = []

      for (const file of list) {
        try {
          const { doc, existed } = await api.uploadDocument(file)
          if (existed) {
            // The record manager matched an existing document: either a no-op
            // duplicate (still ready) or an in-place re-ingest (now pending).
            if (doc.status === 'ready') unchanged.push(doc.filename)
            setDocuments((prev) =>
              prev.some((d) => d.id === doc.id)
                ? prev.map((d) => (d.id === doc.id ? { ...d, ...doc } : d))
                : [doc, ...prev],
            )
          } else {
            uploaded.push(doc)
          }
        } catch (e) {
          const detail = e instanceof Error ? e.message : String(e)
          setError(`${file.name}: ${detail}`)
        }
      }

      void refreshUsage()

      if (unchanged.length > 0)
        showNotice(`${unchanged.join(', ')} unchanged — already ingested`)

      // Realtime will also announce these inserts; the merge in the
      // subscription handler de-duplicates by id, so adding them here is just
      // lower latency.
      if (uploaded.length > 0)
        setDocuments((prev) => [
          ...uploaded.filter((u) => !prev.some((d) => d.id === u.id)),
          ...prev,
        ])
      return uploaded
    },
    [showNotice, refreshUsage],
  )

  const reprocess = useCallback(async (id: string) => {
    setError(null)
    try {
      const doc = await api.reprocessDocument(id)
      setDocuments((prev) => prev.map((d) => (d.id === doc.id ? { ...d, ...doc } : d)))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const reprocessAll = useCallback(async () => {
    setError(null)
    try {
      const { count } = await api.reprocessStale()
      showNotice(`Re-processing ${count} document${count === 1 ? '' : 's'}…`)
      await refreshDocuments()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [refreshDocuments, showNotice])

  const removeDocument = useCallback(async (id: string) => {
    await api.deleteDocument(id)
    setDocuments((prev) => prev.filter((d) => d.id !== id))
    void refreshUsage()
  }, [refreshUsage])

  // The chat is full when its stored message rows reach the server's limit.
  // Counted from what is loaded rather than fetched: this thread's messages are
  // already in memory, so an extra round trip would tell us nothing new.
  const chatFull =
    usage !== null && messages.length >= usage.messages_per_thread_limit

  return {
    threads, activeId, setActiveId, messages, documents, usage, chatFull,
    streaming, streamText, liveTools, error, notice, truncated,
    newThread, removeThread, send, stop, upload, removeDocument,
    reprocess, reprocessAll,
  }
}
