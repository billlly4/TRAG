import { useEffect, useState } from 'react'

import type { DocumentMeta } from '../lib/types'
import { Button, cx } from './ui'

export function Composer({
  documents,
  documentsInThread,
  streaming,
  onSend,
  onStop,
}: {
  documents: DocumentMeta[]
  /** Document ids whose text is already in this thread's context. */
  documentsInThread: Set<string>
  streaming: boolean
  onSend: (text: string, documentIds: string[]) => void
  onStop: () => void
}) {
  const [text, setText] = useState('')
  const [attached, setAttached] = useState<string[]>([])

  // Once a document lands in the thread it is replayed automatically on every
  // later turn, so drop it from the pending selection -- keeping it would send
  // the same text a second time.
  useEffect(() => {
    setAttached((prev) => prev.filter((id) => !documentsInThread.has(id)))
  }, [documentsInThread])

  function submit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || streaming) return
    onSend(trimmed, attached)
    setText('')
    setAttached([])
  }

  function toggle(id: string) {
    setAttached((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    )
  }

  const pendingTokens = documents
    .filter((d) => attached.includes(d.id))
    .reduce((sum, d) => sum + (d.token_estimate ?? 0), 0)

  return (
    <form
      onSubmit={submit}
      className="border-t border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900"
    >
      {documents.length > 0 && (
        <div className="mb-3 flex flex-wrap items-center gap-2">
          {documents.map((d) => {
            const inContext = documentsInThread.has(d.id)
            const isAttached = attached.includes(d.id)
            return (
              <button
                key={d.id}
                type="button"
                disabled={inContext}
                onClick={() => toggle(d.id)}
                title={
                  inContext
                    ? 'Already in this conversation — Claude can still see it'
                    : `Attach to next message${
                        d.token_estimate
                          ? ` · ${d.token_estimate.toLocaleString()} tokens`
                          : ''
                      }`
                }
                className={cx(
                  'rounded-full border px-2.5 py-1 text-xs transition-colors',
                  inContext
                    ? 'cursor-default border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300'
                    : isAttached
                      ? 'border-indigo-500 bg-indigo-50 text-indigo-700 dark:bg-indigo-950/50 dark:text-indigo-300'
                      : 'border-zinc-300 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800',
                )}
              >
                {inContext ? '● ' : isAttached ? '✓ ' : '+ '}
                {d.filename}
              </button>
            )
          })}

          {pendingTokens > 0 && (
            <span className="text-xs text-zinc-400">
              +{pendingTokens.toLocaleString()} tokens this turn
            </span>
          )}
        </div>
      )}

      <div className="flex items-end gap-2">
        <textarea
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) submit(e)
          }}
          placeholder="Ask something…  (Enter to send, Shift+Enter for a newline)"
          className="flex-1 resize-none rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
        />
        {streaming ? (
          <Button type="button" variant="ghost" onClick={onStop}>
            Stop
          </Button>
        ) : (
          <Button type="submit" disabled={!text.trim()}>
            Send
          </Button>
        )}
      </div>
    </form>
  )
}
