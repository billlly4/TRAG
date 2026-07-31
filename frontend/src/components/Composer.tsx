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
  onSend,
  onStop,
}: {
  streaming: boolean
  /** The chat hit its message limit. Readable, but no longer writable. */
  full?: boolean
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
    </form>
  )
}
