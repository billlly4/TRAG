import { useRef, useState } from 'react'

import type { DocumentMeta, Thread } from '../lib/types'
import { Button, cx } from './ui'

export function Sidebar({
  threads,
  activeId,
  documents,
  documentsInThread,
  email,
  onSelect,
  onNew,
  onDeleteThread,
  onUpload,
  onDeleteDocument,
  onSignOut,
}: {
  threads: Thread[]
  activeId: string | null
  documents: DocumentMeta[]
  documentsInThread: Set<string>
  email?: string
  onSelect: (id: string) => void
  onNew: () => void
  onDeleteThread: (id: string) => void
  onUpload: (files: File[]) => Promise<unknown>
  onDeleteDocument: (id: string) => void
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

      <div className="flex-1 overflow-y-auto px-2">
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
        {documents.length === 0 && (
          <p className="px-2 py-2 text-sm text-zinc-400">
            None yet — drag files onto the chat, or use the button below.
          </p>
        )}
        {documents.map((d) => (
          <div key={d.id} className="group flex items-center rounded-lg px-2.5 py-1.5">
            <div className="flex-1 truncate">
              <p className="truncate text-sm text-zinc-700 dark:text-zinc-300">
                {documentsInThread.has(d.id) && (
                  <span
                    title="In this conversation's context"
                    className="mr-1 text-emerald-600 dark:text-emerald-400"
                  >
                    ●
                  </span>
                )}
                {d.filename}
              </p>
              {d.token_estimate != null && (
                <p className="text-xs text-zinc-400">
                  {d.token_estimate.toLocaleString()} tokens
                </p>
              )}
            </div>
            <button
              onClick={() => onDeleteDocument(d.id)}
              title="Delete document"
              className="px-1 text-zinc-400 opacity-0 group-hover:opacity-100 hover:text-red-600"
            >
              ×
            </button>
          </div>
        ))}

        <input
          ref={fileRef}
          type="file"
          hidden
          multiple
          accept=".pdf,.txt,.md,.json,.xml"
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
