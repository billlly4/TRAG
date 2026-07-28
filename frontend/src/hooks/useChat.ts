import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import * as api from '../lib/api'
import { fileIdsInThread } from '../lib/types'
import type { DocumentMeta, Message, Thread } from '../lib/types'

export function useChat() {
  const [threads, setThreads] = useState<Thread[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [documents, setDocuments] = useState<DocumentMeta[]>([])

  const [streaming, setStreaming] = useState(false)
  const [streamText, setStreamText] = useState('')
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
    async (text: string, documentIds: string[]) => {
      let threadId = activeId
      if (!threadId) threadId = (await newThread()).id

      setError(null)
      setTruncated(false)
      setStreaming(true)
      setStreamText('')

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
        documentIds,
        {
          onDelta: (chunk) => setStreamText((prev) => prev + chunk),
          onDone: (frame) => {
            setTruncated(frame.truncated)
            setMessages((prev) => [
              ...prev,
              {
                id: frame.message_id,
                thread_id: threadId,
                role: 'assistant',
                content: frame.content,
                stop_reason: frame.stop_reason,
                usage: frame.usage,
                created_at: new Date().toISOString(),
              },
            ])
            setStreamText('')
            refreshThreads().catch(() => {})
          },
          onError: (detail) => {
            setError(detail)
            setStreamText('')
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

  // Documents whose text is already in this thread's context. Derived from the
  // messages rather than tracked separately, so it stays correct after a
  // reload -- the history is the source of truth.
  const documentsInThread = useMemo(() => {
    const fileIds = fileIdsInThread(messages)
    return new Set(
      documents.filter((d) => fileIds.has(d.anthropic_file_id)).map((d) => d.id),
    )
  }, [messages, documents])

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

    if (uploaded.length > 0) setDocuments((prev) => [...uploaded, ...prev])
    return uploaded
  }, [])

  const removeDocument = useCallback(async (id: string) => {
    await api.deleteDocument(id)
    setDocuments((prev) => prev.filter((d) => d.id !== id))
  }, [])

  return {
    threads, activeId, setActiveId, messages, documents, documentsInThread,
    streaming, streamText, error, truncated,
    newThread, removeThread, send, stop, upload, removeDocument,
  }
}
