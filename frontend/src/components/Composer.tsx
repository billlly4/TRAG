import { useState } from 'react'

import { Button } from './ui'

/**
 * No document picker here any more. Module 1 attached whole documents to a
 * message by hand; retrieval is now automatic -- the assistant searches
 * uploaded documents itself when the question calls for it.
 */
export function Composer({
  streaming,
  full,
  webSearch,
  onWebSearchChange,
  onSend,
  onStop,
}: {
  streaming: boolean
  /** The chat hit its message limit. Readable, but no longer writable. */
  full?: boolean
  /**
   * Whether this message may use web search. Off by default and deliberately
   * per-message rather than a sticky setting: turning the web on is a decision
   * about one question, and a toggle left on last week would silently change
   * where later answers come from.
   */
  webSearch: boolean
  onWebSearchChange: (value: boolean) => void
  onSend: (text: string) => void
  onStop: () => void
}) {
  const [text, setText] = useState('')

  function submit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || streaming || full) return
    onSend(trimmed)
    setText('')
  }

  return (
    <form
      onSubmit={submit}
      className="border-t border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900"
    >
      {/* Said before the send fails, not after. The chat stays fully readable
          -- only new messages are refused. */}
      {full && (
        <p className="mb-2 rounded-lg bg-amber-100 px-3 py-2 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-200">
          This chat is full. Start a new chat to keep going — this one stays
          readable.
        </p>
      )}
      <div className="flex items-end gap-2">
        <textarea
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) submit(e)
          }}
          disabled={full}
          placeholder={
            full
              ? 'This chat has reached its message limit'
              : 'Ask something…  (Enter to send, Shift+Enter for a newline)'
          }
          className="flex-1 resize-none rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none disabled:bg-zinc-100 disabled:text-zinc-400 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100 dark:disabled:bg-zinc-900"
        />
        {streaming ? (
          <Button type="button" variant="ghost" onClick={onStop}>
            Stop
          </Button>
        ) : (
          <Button type="submit" disabled={!text.trim() || full}>
            Send
          </Button>
        )}
      </div>

      {/* The wording matters: without this, an unanswerable question about the
          user's files gets "I don't have that information", and there is no way
          to tell that was a deliberate refusal rather than a broken search. */}
      <label className="mt-2 flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
        <input
          type="checkbox"
          checked={webSearch}
          onChange={(e) => onWebSearchChange(e.target.checked)}
          disabled={full}
          className="h-3.5 w-3.5 rounded border-zinc-300 text-indigo-600 focus:ring-indigo-500 dark:border-zinc-600 dark:bg-zinc-800"
        />
        <span>
          Search the web for this message
          <span className="ml-1 text-zinc-400 dark:text-zinc-500">
            — off, answers come only from your documents
          </span>
        </span>
      </label>
    </form>
  )
}
