import { useRef, useState } from 'react'

import type { DocumentMeta, Thread } from '../lib/types'
import { Banner, Button, cx } from './ui'

/**
 * Ingestion status badge. The transitions arrive over Supabase Realtime as the
 * backend walks pending -> extracting -> chunking -> embedding -> ready, so
 * this advances live without a refresh.
 */
function StatusBadge({ doc }: { doc: DocumentMeta }) {
  if (doc.status === 'ready') {
    return (
      <p className="text-xs text-emerald-600 dark:text-emerald-400">
        ready · {doc.chunk_count ?? 0} chunks
        {doc.stale && (
          <span
            className="text-amber-600 dark:text-amber-400"
            title="Chunks were built under older settings — re-process to update"
          >
            {' '}
            · config changed
          </span>
        )}
      </p>
    )
  }
  if (doc.status === 'failed') {
    return (
      <p
        className="truncate text-xs text-red-600 dark:text-red-400"
        title={doc.error ?? undefined}
      >
        failed{doc.error ? ` · ${doc.error}` : ''}
      </p>
    )
  }
  return (
    <p className="text-xs text-zinc-400">
      <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500 align-middle" />
      {doc.status}…
    </p>
  )
}

export function Sidebar({
  threads,
  activeId,
  documents,
  notice,
  email,
  onSelect,
  onNew,
  onDeleteThread,
  onUpload,
  onDeleteDocument,
  onReprocess,
  onReprocessAll,
  onSignOut,
}: {
  threads: Thread[]
  activeId: string | null
  documents: DocumentMeta[]
  notice: string | null
  email?: string
  onSelect: (id: string) => void
  onNew: () => void
  onDeleteThread: (id: string) => void
  onUpload: (files: File[]) => Promise<unknown>
  onDeleteDocument: (id: string) => void
  onReprocess: (id: string) => void
  onReprocessAll: () => void
  onSignOut: () => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)

  async function pick(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? [])
    if (files.length === 0) return
    setUploading(true)
    await onUpload(files)
    setUploading(false)
    e.target.value = ''
  }

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="p-3">
        <Button className="w-full" onClick={onNew}>
          New chat
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2">
        <p className="px-2 py-1 text-xs font-medium tracking-wide text-zinc-400 uppercase">
          Threads
        </p>
        {threads.length === 0 && (
          <p className="px-2 py-2 text-sm text-zinc-400">No threads yet</p>
        )}
        {threads.map((t) => (
          <div
            key={t.id}
            className={cx(
              'group flex items-center rounded-lg',
              t.id === activeId
                ? 'bg-zinc-200 dark:bg-zinc-800'
                : 'hover:bg-zinc-100 dark:hover:bg-zinc-900',
            )}
          >
            <button
              onClick={() => onSelect(t.id)}
              className="flex-1 truncate px-2.5 py-2 text-left text-sm text-zinc-700 dark:text-zinc-300"
            >
              {t.title || 'Untitled'}
            </button>
            <button
              onClick={() => onDeleteThread(t.id)}
              title="Delete thread"
              className="px-2 text-zinc-400 opacity-0 group-hover:opacity-100 hover:text-red-600"
            >
              ×
            </button>
          </div>
        ))}

        <p className="mt-4 px-2 py-1 text-xs font-medium tracking-wide text-zinc-400 uppercase">
          Documents
        </p>

        {notice && (
          <div className="px-1 py-1">
            <Banner tone="info">{notice}</Banner>
          </div>
        )}

        {(() => {
          // Docs whose chunks predate the current config and can be rebuilt
          // (mid-pipeline ones are already being rebuilt).
          const staleCount = documents.filter(
            (d) => d.stale && (d.status === 'ready' || d.status === 'failed'),
          ).length
          return staleCount > 0 ? (
            <div className="px-1 pb-1">
              <Button variant="ghost" className="w-full" onClick={onReprocessAll}>
                ↻ Re-process {staleCount} stale
              </Button>
            </div>
          ) : null
        })()}

        {documents.length === 0 && (
          <p className="px-2 py-2 text-sm text-zinc-400">
            None yet — drag files onto the chat, or use the button below.
          </p>
        )}
        {documents.map((d) => {
          const terminal = d.status === 'ready' || d.status === 'failed'
          // Terminal + stale -> the normal "config changed" re-process.
          // Non-terminal -> recovery. An ingest orphaned by a backend crash or
          // restart sits at pending/extracting forever with nothing working on
          // it, and looks identical to one still in progress: there is no
          // status-changed timestamp to tell them apart. So the button is
          // offered for every in-flight document and the tooltip says what it
          // is for. Clicking it on a genuinely running ingest just restarts
          // that ingest.
          const canReprocess = terminal ? d.stale : true
          return (
            <div key={d.id} className="group flex items-center rounded-lg px-2.5 py-1.5">
              {/* The extracted title goes in the tooltip rather than the row:
                  it is often far longer than the filename and would push the
                  status badge out of view. */}
              <div className="flex-1 truncate" title={d.title ?? undefined}>
                <p className="truncate text-sm text-zinc-700 dark:text-zinc-300">
                  {d.filename}
                </p>
                {(d.doc_type || d.published_year) && (
                  <p className="truncate text-xs text-zinc-400">
                    {[d.doc_type, d.published_year].filter(Boolean).join(' · ')}
                  </p>
                )}
                <StatusBadge doc={d} />
              </div>
              {canReprocess && (
                <button
                  onClick={() => onReprocess(d.id)}
                  title={
                    terminal
                      ? 'Re-process under current settings'
                      : 'Stuck? Re-process from the stored file'
                  }
                  className="px-1 text-zinc-400 opacity-0 group-hover:opacity-100 hover:text-indigo-600"
                >
                  ↻
                </button>
              )}
              <button
                onClick={() => onDeleteDocument(d.id)}
                title="Delete document"
                className="px-1 text-zinc-400 opacity-0 group-hover:opacity-100 hover:text-red-600"
              >
                ×
              </button>
            </div>
          )
        })}

        <input
          ref={fileRef}
          type="file"
          hidden
          multiple
          accept=".pdf,.docx,.html,.md,.txt"
          onChange={pick}
        />
        <Button
          variant="ghost"
          className="mt-1 w-full"
          disabled={uploading}
          onClick={() => fileRef.current?.click()}
        >
          {uploading ? 'Uploading…' : '+ Upload document'}
        </Button>
      </div>

      <div className="border-t border-zinc-200 p-3 dark:border-zinc-800">
        <p className="mb-2 truncate text-xs text-zinc-500">{email}</p>
        <Button variant="ghost" className="w-full" onClick={onSignOut}>
          Sign out
        </Button>
      </div>
    </aside>
  )
}
